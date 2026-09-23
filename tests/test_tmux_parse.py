"""Tests for tmux pane output parsing (pure functions).

Fixtures are real capture-pane excerpts from live spikes against
Claude Code 2.1.168. These functions must NOT touch tmux/subprocess —
they only parse text, so they are fully unit-testable.

Key invariant (from blind review + spikes): the model's answer markers are
LINE-ANCHORED (on their own line); the input echo shows the markers inline
(mid-sentence). So line-anchored matching alone distinguishes answer from
echo and from the model quoting the marker syntax inside its reply.
"""

from pathlib import Path

import pytest

from d_brain.services.tmux_parse import (
    _WORKING_RE,
    PaneState,
    classify_state,
    extract_open_reply,
    extract_reply,
    find_latest_reply,
    find_pending_replies,
    find_unhandled_replies,
    foreign_view_label,
    has_marker,
    input_box_text,
    is_agents_list_view,
    is_complete,
    is_main_turn_active,
    is_working,
    main_area_working,
    main_turn_finished,
    open_reply_rids,
    parse_reset_time,
    reply_rids,
    reset_epoch,
    strip_open_reply_body,
    strip_reply_bodies,
    strip_right_column,
    unwrap_soft_breaks,
)

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ── Real capture excerpts (claude 2.1.168) ──────────────────────────────

READY_CAPTURE = """\
╭─── Claude Code v2.1.168 ──────────────────────────────────────╮
│                Welcome back Majento!                          │
╰───────────────────────────────────────────────────────────────╯
────────────────────────────────────────────────────────────────
❯
────────────────────────────────────────────────────────────────
  hello | Opus 4.8 (1M context) | ~/T/dbrain_spk
  ⏵⏵ bypass permissions on (shift+tab to cycle) · ← for agents
"""

# READY but in a normal permission mode (no "bypass permissions" footer);
# only the idle ❯ with a ghost suggestion is present. Exercises the ❯ branch.
READY_NO_BYPASS_CAPTURE = """\
────────────────────────────────────────────────────────────────
❯ Try "fix the failing test"
────────────────────────────────────────────────────────────────
  hello | Opus 4.8 (1M context) | ~/project
"""

TRUST_CAPTURE = """\
 Accessing workspace:
 /private/tmp
 Quick safety check: Is this a project you created or one you trust?
 Claude Code'll be able to read, edit, and execute files here.
 ❯ 1. Yes, I trust this folder
   2. No, exit
 Enter to confirm · Esc to cancel
"""

# Verbatim from the binary (v2.1.179): title, body and the numbered menu shown
# on first run with --dangerously-skip-permissions on a fresh config dir.
BYPASS_CAPTURE = """\
 WARNING: Claude Code running in Bypass Permissions mode

 In Bypass Permissions mode, Claude Code will not ask for your approval before
 running potentially dangerous commands.
 This mode should only be used in a sandboxed container/VM that has restricted
 internet access and can easily be restored if damaged.

 ❯ 1. No, exit
   2. Yes, I accept
"""

STARTING_CAPTURE = """\
╭─── Claude Code v2.1.168 ──────────────────────────────────────╮
│                  Loading...                                    │
╰───────────────────────────────────────────────────────────────╯
"""

# NOTE: rate-limit / logged-out strings NOT yet observed live — synthetic,
# based on known Claude Code wording. Must be verified on the VPS (open Q #2).
RATE_LIMIT_CAPTURE = """\
  You've reached your usage limit. Your limit resets at 3:00 PM.
❯
  ⏵⏵ bypass permissions on (shift+tab to cycle)
"""

LOGGED_OUT_CAPTURE = """\
  Invalid API key · Please run /login to authenticate.
❯
"""


# ── extract_reply ───────────────────────────────────────────────────────


def test_extract_reply_takes_line_anchored_pair_not_inline_echo():
    """Input echo has markers inline (mid-line); the answer has them on
    their own lines. Only the line-anchored pair is the real reply."""
    rid = "911e06a2"
    text = (
        f"> Reply PONG, put <<<R:{rid}>>> before and <<<E:{rid}>>> after\n"  # echo
        "  ... transcript ...\n"
        f"<<<R:{rid}>>>\n"
        "PONG\n"
        f"<<<E:{rid}>>>\n"
        "❯\n"
    )
    assert extract_reply(text, rid) == "PONG"


def test_extract_reply_with_tui_bullet_prefix():
    """Regression (live): Claude Code prefixes the first answer line with
    '⏺ ' and indents the rest. The marker is at the END of its line; that,
    not 'starts the line', is what distinguishes answer from inline echo."""
    rid = "bull1234"
    text = f"⏺ <<<R:{rid}>>>\n  PONG\n  <<<E:{rid}>>>\n❯\n"
    assert extract_reply(text, rid) == "PONG"


def test_extract_reply_multiline_answer():
    rid = "abc12345"
    text = f"<<<R:{rid}>>>\nline one\nline two\n<<<E:{rid}>>>\n"
    assert extract_reply(text, rid) == "line one\nline two"


def test_extract_reply_marker_quoted_inline_inside_answer():
    """CRITICAL (blind review): the model quotes the marker syntax inside
    its answer (inline). The inline quote must be ignored; the real
    line-anchored pair still yields the full answer."""
    rid = "deadbeef"
    text = (
        f"<<<R:{rid}>>>\n"
        f"To finish I print <<<E:{rid}>>> on its own line.\n"  # inline quote
        "Here is the real answer.\n"
        f"<<<E:{rid}>>>\n"
        "❯\n"
    )
    got = extract_reply(text, rid)
    assert got is not None
    assert "Here is the real answer." in got
    assert got.startswith("To finish I print")


def test_extract_reply_stray_end_marker_does_not_corrupt():
    """HIGH (blind review): a stray line-anchored end marker after a complete
    pair must not produce a span that swallows the real end marker + chrome."""
    rid = "cafe1234"
    text = (
        f"<<<R:{rid}>>>\n"
        "PONG\n"
        f"<<<E:{rid}>>>\n"
        "❯\n"
        f"<<<E:{rid}>>>\n"  # stray, no preceding R
    )
    got = extract_reply(text, rid)
    assert got == "PONG"


def test_extract_reply_none_when_no_markers():
    assert extract_reply("just some text\n❯\n", "deadbeef") is None


def test_extract_reply_none_when_only_open_marker():
    rid = "feedface"
    assert extract_reply(f"<<<R:{rid}>>>\nPONG (still typing)", rid) is None


def test_extract_reply_ignores_other_rid():
    text = "<<<R:aaaa1111>>>\nPONG\n<<<E:aaaa1111>>>\n"
    assert extract_reply(text, "bbbb2222") is None


def test_extract_reply_empty_rid_raises():
    with pytest.raises(ValueError):
        extract_reply("anything", "")


# ── find_latest_reply ────────────────────────────────────────────────────


def test_find_latest_reply_finds_unknown_rid():
    rid = "auto1234"
    text = f"<<<R:{rid}>>>\nDone in background.\n<<<E:{rid}>>>\n❯\n"
    assert find_latest_reply(text) == (rid, "Done in background.")


def test_find_latest_reply_picks_last_pair_when_several_present():
    text = (
        "<<<R:first111>>>\nold reply\n<<<E:first111>>>\n"
        "❯\n"
        "<<<R:second22>>>\nnew reply\n<<<E:second22>>>\n"
    )
    assert find_latest_reply(text) == ("second22", "new reply")


def test_find_latest_reply_ignores_inline_echo():
    rid = "echo1234"
    text = f"> reply between <<<R:{rid}>>> and <<<E:{rid}>>> markers\n"
    assert find_latest_reply(text) is None


def test_find_latest_reply_none_when_no_markers():
    assert find_latest_reply("just some text\n❯\n") is None


def test_find_latest_reply_none_on_dangling_open_marker():
    assert find_latest_reply("<<<R:pending1>>>\nstill typing") is None


def test_find_latest_reply_skips_stray_end_marker_falling_back_to_real_pair():
    """A stray end marker for a rid that never opened must be skipped, not
    mistaken for a valid (rid, body) pair."""
    text = "<<<E:orphan01>>>\n<<<R:real0001>>>\nactual reply\n<<<E:real0001>>>\n"
    assert find_latest_reply(text) == ("real0001", "actual reply")


# ── is_complete ─────────────────────────────────────────────────────────


def test_is_complete_true_on_line_anchored_pair():
    rid = "11112222"
    assert is_complete(f"<<<R:{rid}>>>\nPONG\n<<<E:{rid}>>>\n", rid) is True


def test_is_complete_false_on_inline_echo_only():
    """The echo of the typed prompt (inline markers) is NOT a complete reply."""
    rid = "33334444"
    text = f"> do thing, wrap in <<<R:{rid}>>> .. <<<E:{rid}>>>\n❯\n"
    assert is_complete(text, rid) is False


def test_is_complete_false_when_answer_still_streaming():
    rid = "55556666"
    assert is_complete(f"<<<R:{rid}>>>\npartial...", rid) is False


# ── classify_state ──────────────────────────────────────────────────────


def test_classify_trust_prompt():
    assert classify_state(TRUST_CAPTURE) == PaneState.TRUST_PROMPT


def test_classify_bypass_prompt():
    assert classify_state(BYPASS_CAPTURE) == PaneState.BYPASS_PROMPT


def test_classify_bypass_not_triggered_by_prose():
    # A model REPLY that merely describes bypass mode must not be classified as
    # the accept screen — the title anchor isn't present.
    prose = (
        "────────────────────\n"
        "⏺ Bypass Permissions mode means Claude Code will not ask for approval.\n"
        "  To accept it you would choose 'Yes, I accept'.\n"
        "────────────────────\n❯\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    assert classify_state(prose) == PaneState.READY


def test_classify_bypass_not_triggered_by_numbered_list_reply():
    # The dangerous false-positive: a reply that emits a numbered consent list
    # whose line 2 reads "2. Yes, I accept …". Without the verbatim warning
    # TITLE it must NOT be taken for the accept screen (else the watchdog's
    # current_state() would inject "2" into a live session).
    numbered = (
        "────────────────────\n"
        "⏺ Pick one:\n"
        "  1. No, decline\n"
        "  2. Yes, I accept the terms\n"
        "────────────────────\n❯\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    assert classify_state(numbered) == PaneState.READY


def test_classify_bypass_ignored_when_idle_footer_present():
    # Codex scenario: a completed reply quotes BOTH the verbatim warning title
    # AND a "2. Yes, I accept" line, sitting in scrollback while the session is
    # idle (capture spans -S -200). The live idle footer is present, so this is
    # a healthy READY session, NOT the modal — the real modal never co-exists
    # with the footer. Misclassifying it would let the watchdog (BYPASS_PROMPT
    # is not serviceable) force-recover a healthy session.
    quoted = (
        "⏺ The first-run screen reads:\n"
        " WARNING: Claude Code running in Bypass Permissions mode\n"
        "   1. No, exit\n"
        "   2. Yes, I accept\n"
        "────────────────────\n❯\n────────────────────\n"
        "  hello | Opus 4.8 (1M context) | ~/p\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    assert classify_state(quoted) == PaneState.READY


def test_classify_ready_with_bypass_footer():
    assert classify_state(READY_CAPTURE) == PaneState.READY


def test_classify_ready_without_bypass_footer():
    """READY must be detected via the idle ❯ alone (bypass footer is
    permission-mode specific and may be absent)."""
    assert classify_state(READY_NO_BYPASS_CAPTURE) == PaneState.READY


def test_classify_ready_when_footer_above_empty_bottom():
    """Regression (live): the TUI draws content at the top and leaves the
    bottom of the screen blank, so the footer (bypass/idle) is NOT in the
    chrome region. READY must still be detected via the footer anchor."""
    text = READY_CAPTURE + "\n" * 30
    assert classify_state(text) == PaneState.READY


def test_classify_starting():
    assert classify_state(STARTING_CAPTURE) == PaneState.STARTING


def test_classify_rate_limited():
    assert classify_state(RATE_LIMIT_CAPTURE) == PaneState.RATE_LIMITED


def test_classify_logged_out():
    assert classify_state(LOGGED_OUT_CAPTURE) == PaneState.LOGGED_OUT


def test_classify_unknown_on_empty():
    assert classify_state("") == PaneState.UNKNOWN


def test_rate_limit_takes_priority_over_ready_idle():
    """A rate-limit banner co-existing with idle must classify as
    RATE_LIMITED so the watchdog does NOT treat it as healthy."""
    assert classify_state(RATE_LIMIT_CAPTURE) == PaneState.RATE_LIMITED


# ── classify_state false positives (transcript body mentions triggers) ───


def _long_transcript(mention: str) -> str:
    """A long transcript whose BODY mentions a trigger word, but whose
    chrome region (bottom) is a healthy idle pane."""
    body = f"❯ explain something\nThe model says: {mention}\n" + "filler line\n" * 25
    footer = (
        "────────────────────────────────────────────\n"
        "❯\n"
        "────────────────────────────────────────────\n"
        "  hello | Opus 4.8 (1M context) | ~/project\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    return body + footer


def test_classify_no_false_positive_rate_limit_in_body():
    assert (
        classify_state(_long_transcript("you will hit your usage limit soon"))
        == PaneState.READY
    )


def test_classify_no_false_positive_login_in_body():
    assert (
        classify_state(_long_transcript("just run /login to authenticate"))
        == PaneState.READY
    )


def test_classify_no_false_positive_trust_in_body():
    assert (
        classify_state(
            _long_transcript("Is this a project you trust? appears on first run")
        )
        == PaneState.READY
    )


def test_classify_trust_when_menu_above_empty_chrome():
    """Regression (found in live integration): the trust dialog is a
    full-screen modal drawn at the TOP; on a tall pane the bottom 18 lines
    (chrome) are blank. Trust must still be detected over the whole pane."""
    text = TRUST_CAPTURE + "\n" * 40
    assert classify_state(text) == PaneState.TRUST_PROMPT


def test_classify_priority_stack_trust_first():
    """When several signatures co-occur in the chrome, TRUST wins (it must be
    answered before anything else is meaningful)."""
    stacked = (
        " Is this a project you created or one you trust?\n"
        " ❯ 1. Yes, I trust this folder\n"
        "   2. No, exit\n"
        " usage limit resets at 3:00 PM\n"
        " ❯\n"
    )
    assert classify_state(stacked) == PaneState.TRUST_PROMPT


def test_extract_reply_works_for_skill_invocation_turn():
    """Characterization: a /skill-name prompt is a NORMAL model turn — the
    model honors the appended marker instruction, so the existing marker path
    extracts the reply. No verbatim extractor is needed for skills."""
    rid = "ab12cd34"
    pane = (
        "❯ /vault-note сохрани мысль про autograph\n"
        "\n"
        "  When done, wrap your ENTIRE reply between a line containing only "
        f"<<<R:{rid}>>> and a line containing only <<<E:{rid}>>>.\n"
        "\n"
        f"⏺ <<<R:{rid}>>>\n"
        "  Заметка сохранена: thoughts/ideas/autograph.md\n"
        f"  <<<E:{rid}>>>\n"
        "\n"
        "❯\n"
    )
    assert is_complete(pane, rid)
    assert extract_reply(pane, rid) == "Заметка сохранена: thoughts/ideas/autograph.md"


# ── is_idle (turn-completion signal independent of the bypass footer) ──────

_FOOTER = (
    "────────────────────\n❯\n────────────────────\n"
    "  hello | Opus 4.8 (1M context) | ~/p\n"
    "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
)
_WORKING = "  ✻ Working…  (esc to interrupt)\n"


def test_is_idle_true_on_empty_input_prompt():
    from d_brain.services.tmux_parse import is_idle

    assert is_idle("transcript above\n" + _FOOTER)


def test_is_idle_false_while_working():
    from d_brain.services.tmux_parse import is_idle

    assert not is_idle("transcript\n" + _WORKING + _FOOTER)


def test_is_idle_false_when_footer_present_but_thinking():
    """The bypass footer is ALWAYS on screen under --dangerously-skip-
    permissions, so it must never be treated as an idle signal by itself."""
    from d_brain.services.tmux_parse import is_idle

    footer = "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    pane = "long transcript\n" + _WORKING + footer
    assert not is_idle(pane)


def test_is_idle_false_on_empty_pane():
    from d_brain.services.tmux_parse import is_idle

    assert not is_idle("")


def test_is_idle_false_on_menu_selector():
    """An interactive menu's selector (`❯ 1. Yes …`) is NOT an idle prompt —
    only a bare ❯ (empty input line) counts. Guards wrap=False completion
    against approval/menu prompts."""
    from d_brain.services.tmux_parse import is_idle

    pane = (
        "Do you approve this plan?\n"
        " ❯ 1. Yes, proceed\n   2. No, keep planning\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    assert not is_idle(pane)


def test_classify_login_menu_as_logged_out():
    """Incident 2026-06-10: a fresh CLAUDE_CONFIG_DIR sent the new process
    into first-run onboarding (theme → login menu); classify_state saw
    UNKNOWN and the watchdog stayed silent while every ask timed out. The
    login/onboarding screens need a human — classify them LOGGED_OUT."""
    login = (
        " Claude Code can be used with your Claude subscription or billed "
        "based on API usage through your Console account.\n"
        " Select login method:\n"
        " ❯ 1. Claude account with subscription · Pro, Max, Team, or Enterprise\n"
        "   2. Anthropic Console account · API usage billing\n"
    )
    assert classify_state(login) == PaneState.LOGGED_OUT


def test_classify_onboarding_theme_as_logged_out():
    theme = (
        "   3. Light mode\n ❯ 6. Dark mode (ANSI colors only) ✔\n"
        "  Syntax theme: ansi (ctrl+t to disable)\n"
    )
    assert classify_state(theme) == PaneState.LOGGED_OUT


def test_classify_first_run_security_notes_as_logged_out():
    """Clean-server rehearsal: the bot's session (CLAUDE_CONFIG_DIR=~/.claude)
    stopped on this first-run screen and every canary timed out silently."""
    notes = (
        " Security notes:\n"
        " 1. Claude can make mistakes.\n"
        "    You're responsible for Claude's actions and should always\n"
        "    review them, especially when running code.\n"
        " 2. Due to prompt injection risks, only use it with code you trust\n"
        "    Learn more: https://code.claude.com/docs/en/security\n\n"
        " Press Enter to continue…\n"
    )
    assert classify_state(notes) == PaneState.LOGGED_OUT


def test_survey_prompt_detected():
    from d_brain.services.tmux_parse import has_survey_prompt

    pane = (
        "● How is Claude doing this session? (optional)\n"
        "  1: Bad    2: Fine   3: Good   0: Dismiss\n"
        "❯\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    assert has_survey_prompt(pane)
    assert not has_survey_prompt("❯\n  ⏵⏵ bypass permissions on\n")


def test_is_working_detects_new_spinner_format():
    # Claude Code dropped the "esc to interrupt" hint; the live turn now
    # shows a spinner with an elapsed-time + token counter.
    from d_brain.services.tmux_parse import is_working

    assert is_working("reply so far\n✢ Razzle-dazzling… (44s · ↓1.8k tokens)\n")
    assert is_working("✽ Booping… (55s · ↓4.1k tokens)\n")


def test_is_working_still_detects_legacy_hint():
    from d_brain.services.tmux_parse import is_working

    assert is_working("  ✻ Working…  (esc to interrupt)\n")


def test_is_working_false_at_idle():
    from d_brain.services.tmux_parse import is_working

    assert not is_working("❯\n  ⏵⏵ bypass permissions on (shift+tab to cycle)\n")


# ── 2026-08-20: state signatures must not fire on the model's own text ───
#
# The brain's pane had drifted to 80x23, so the bottom-18-line "chrome"
# window covered the model's answer. Three times in one day a reply that
# merely discussed rate limits reported the session as rate limited and the
# user got "⏳ Лимит подписки исчерпан" while nothing was wrong.

_FOOTER = (
    "────────────────────────────────────────────────\n"
    "❯\n"
    "────────────────────────────────────────────────\n"
    "  hello | Opus 4.8 | ~/vault\n"
    "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
)


def test_reply_discussing_limits_is_not_rate_limited():
    """The exact false positive: an answer about the limit bug, quoting the
    banner, sits in the chrome window on a short pane."""
    text = (
        "<<<R:talk0001>>>\n"
        "  Разобрался: строка «You've hit your session limit · resets 12am\n"
        "  (UTC)» — это usage limit reached в панели, а не реальный лимит.\n"
        "<<<E:talk0001>>>\n" + _FOOTER
    )
    assert classify_state(text) == PaneState.READY


def test_real_limit_banner_outside_markers_still_detected():
    text = (
        "<<<R:prev0001>>>\nОбычный ответ без ключевых слов.\n<<<E:prev0001>>>\n"
        "  Claude usage limit reached. Your limit will reset at 3pm (UTC).\n❯\n"
    )
    assert classify_state(text) == PaneState.RATE_LIMITED


def test_soft_limit_warning_is_not_a_block():
    """ "Approaching your weekly limit" is a heads-up; the session still
    works and must not be reported as exhausted."""
    text = "  Heads up: you are approaching your weekly limit.\n" + _FOOTER
    assert classify_state(text) == PaneState.READY


def test_casual_mention_of_rate_limit_is_not_a_block():
    """Tool output (an API gateway schema has a rate_limit column) and ordinary
    prose used to match the old pattern verbatim."""
    text = "  column: rate_limit_per_min — usage limit for the tenant\n" + _FOOTER
    assert classify_state(text) == PaneState.READY


def test_strip_reply_bodies_keeps_unterminated_span():
    """A limit banner can appear DURING a turn, i.e. after an open marker
    with no close yet — that text must survive stripping."""
    text = "<<<R:live0001>>>\nStill writing… usage limit reached\n"
    assert "usage limit reached" in strip_reply_bodies(text)


def test_strip_reply_bodies_preserves_a_real_banner_above_a_closed_span():
    """Regression (found re-driving the consolidated plan, 2026-08-20): the
    span regex used to be compiled with re.S over its WHOLE pattern, which
    let the leading `^.*?` cross newlines too. Because multiline `^` already
    matches position 0, the lazy prefix never needed to restart at the
    marker's own line — it swallowed every line ABOVE the open marker,
    including a real rate-limit banner sitting higher in the pane, deleting
    the exact text strip_reply_bodies exists to protect. A banner above a
    fully closed span, and chrome below it, must both survive."""
    text = (
        "line one CLI banner\n"
        "usage limit reached - resets 12am\n"
        "line three\n"
        "<<<R:abc123>>>\n"
        "reply body across\n"
        "multiple lines\n"
        "<<<E:abc123>>>\n"
        "tail chrome\n"
    )
    out = strip_reply_bodies(text)
    assert "usage limit reached - resets 12am" in out
    assert "tail chrome" in out
    assert "reply body across" not in out


def test_strip_reply_bodies_strips_multiple_spans_in_sequence():
    text = (
        "pre\n"
        "<<<R:a1>>>\nreply one\n<<<E:a1>>>\n"
        "mid\n"
        "<<<R:a2>>>\nreply two\n<<<E:a2>>>\n"
        "post\n"
    )
    out = strip_reply_bodies(text)
    assert "pre" in out and "mid" in out and "post" in out
    assert "reply one" not in out
    assert "reply two" not in out


# ── is_working: the modern TUI's active-turn signatures ──────────────────


def test_waiting_for_background_agent_counts_as_working():
    """Measured live 2026-08-20 mid-turn: while the turn waits on a subagent
    the TUI shows neither 'esc to interrupt' nor a parenthesised spinner, so
    is_working() said False on a demonstrably live turn — that is how
    6-9 minute tool-dense turns tripped the stall detector."""
    text = (
        "✻ Waiting for 1 background agent to finish\n"
        "────────────────────────────────────────────────\n❯\n"
        "  ⏵⏵ bypass permissions on · 1 monitor\n"
    )
    assert is_working(text) is True


def test_agent_row_elapsed_time_counts_as_working():
    text = "  ◍ general-purpose  Inspecting panes 3m 5s · ↓ 88.3k tokens\n❯\n"
    assert is_working(text) is True


def test_legacy_spinner_still_counts_as_working():
    assert is_working("  ✻ Working…  (esc to interrupt)\n") is True
    assert is_working("  ✢ Razzle-dazzling… (44s · ↓1.8k tokens)\n") is True


def test_plain_idle_pane_is_not_working():
    assert is_working(READY_CAPTURE) is False


def test_arbitrary_tool_output_does_not_count_as_working():
    """2026-08-20 review, reproduced live: dropping the "(" anchor to catch
    the bare "2m 15s ·" spinner form also made the elapsed-time clause match
    ANY line shaped like "<Ns> ·" — including tool output that has nothing
    to do with the spinner. The CLI's own readout is always paired with its
    token-count arrow; requiring "· ↓" keeps the real forms matching (see
    test_agent_row_elapsed_time_counts_as_working /
    test_legacy_spinner_still_counts_as_working) while excluding this."""
    assert is_working("  tool ran 12s · ok\n❯\n") is False


# ── is_working_progressing: change-aware liveness (review 2026-08-20) ────
#
# is_working() alone reports True on a genuinely FROZEN frame (a dead
# background subagent, a process stuck in D-state) that happens to match the
# "waiting for N background agents" / bare elapsed-timer signature — those
# are the ONLY two signatures expected to tick over poll-to-poll; requiring
# change closes the false-positive without touching "esc to interrupt",
# which a real silent-but-alive turn holds static for its whole duration.


def test_is_working_progressing_background_agent_needs_change():
    from d_brain.services.tmux_parse import is_working_progressing

    frozen = (
        "✻ Waiting for 1 background agent to finish\n"
        "  agent  3m 5s · ↓ 88.3k tokens\n❯\n"
    )
    assert is_working_progressing(frozen, None) is True  # no baseline yet
    assert is_working_progressing(frozen, frozen) is False  # unchanged
    later = (
        "✻ Waiting for 1 background agent to finish\n"
        "  agent  3m 9s · ↓ 88.3k tokens\n❯\n"
    )
    assert is_working_progressing(later, frozen) is True  # elapsed ticked


def test_is_working_progressing_elapsed_timer_needs_change():
    from d_brain.services.tmux_parse import is_working_progressing

    frame = "✢ Razzle-dazzling… (44s · ↓1.8k tokens)\n"
    prev = "✢ Razzle-dazzling… (43s · ↓1.8k tokens)\n"
    assert is_working_progressing(frame, frame) is False
    assert is_working_progressing(frame, prev) is True


def test_is_working_progressing_legacy_hint_exempt_from_change():
    from d_brain.services.tmux_parse import is_working_progressing

    static = "  ✻ Working…  (esc to interrupt)\n"
    assert is_working_progressing(static, static) is True
    assert is_working_progressing(static, None) is True


# ── trust_static: bounded trust in the legacy hint ─────
#
# "esc to interrupt" lives in a footer this CLI shows whenever ANYTHING is
# interruptible (measured: 99.6% of its occurrences on a real pane.log), so
# trusting it forever on a byte-identical frame disarms hang detection. The
# caller owns the clock and decides WHEN trust runs out; this function only
# honours the decision. Default must stay the old behaviour — every existing
# call site depends on it.


def test_is_working_progressing_static_untrusted_needs_change():
    """Regression: with trust_static=False a byte-identical STATIC
    frame is no longer liveness — it must obey the same "chrome changed"
    rule the PROGRESS signatures already live under."""
    from d_brain.services.tmux_parse import is_working_progressing

    static = "  ✻ Working…  (esc to interrupt)\n"
    assert is_working_progressing(static, static, trust_static=False) is False


def test_is_working_progressing_static_trusted_by_default():
    """The SAME frame with no parameter passed stays True — this is the
    guard for every existing call site (and for a legitimately silent long
    turn, which holds the hint byte-identical for its whole duration)."""
    from d_brain.services.tmux_parse import is_working_progressing

    static = "  ✻ Working…  (esc to interrupt)\n"
    assert is_working_progressing(static, static) is True


def test_is_working_progressing_static_untrusted_still_true_when_changed():
    """trust_static=False withdraws the exemption, not the signature: a
    STATIC frame that DID change since the previous poll is still live."""
    from d_brain.services.tmux_parse import is_working_progressing

    prev = "  ✻ Working…  (esc to interrupt)\n"
    now = "  ✻ Working…  (esc to interrupt)\n  read foo.py\n"
    assert is_working_progressing(now, prev, trust_static=False) is True
    assert is_working_progressing(now, None, trust_static=False) is True


def test_is_working_progressing_false_at_idle():
    from d_brain.services.tmux_parse import is_working_progressing

    idle = "❯\n  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    assert is_working_progressing(idle, idle) is False


# ── parse_reset_time ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "banner,expected",
    [
        ("You've hit your session limit · resets 12am (UTC)", (0, 0)),
        ("Claude usage limit reached. Your limit will reset at 3pm (UTC)", (15, 0)),
        ("session limit reached · resets at 15:30 (UTC)", (15, 30)),
    ],
)
def test_parse_reset_time_reads_the_banner(banner, expected):
    assert parse_reset_time(f"  {banner}\n❯\n") == expected


def test_parse_reset_time_none_without_a_limit_banner():
    assert parse_reset_time(READY_CAPTURE) is None


def test_parse_reset_time_none_when_banner_has_no_time():
    assert parse_reset_time("  Claude usage limit reached.\n❯\n") is None


@pytest.mark.parametrize(
    "banner",
    [
        "5-hour limit reached · resets 12pm",
        "weekly limit reached · resets 9am",
    ],
)
def test_parse_reset_time_none_without_utc_marker(banner):
    """The watchdog treats the returned (hour, minute) as UTC unconditionally
    (2026-08-20 review): a banner with no "(UTC)"/"UTC" on the same line must
    be treated as unparseable rather than silently assumed to be UTC — the
    CLI's rendering has changed format before, and a wrong-timezone guess
    with no log line to explain it is worse than falling back to
    limit_max_wait."""
    assert parse_reset_time(f"  {banner}\n❯\n") is None


def test_parse_reset_time_requires_utc_on_the_same_line():
    """A stray "UTC" elsewhere on screen must not launder an unmarked reset
    time — only a marker on the banner's OWN line counts."""
    text = "  Some other line mentions UTC.\n  weekly limit reached · resets 9am\n❯\n"
    assert parse_reset_time(text) is None


# ── unmarked_reset_banner: diagnostic companion to parse_reset_time ──────


def test_unmarked_reset_banner_returns_the_line_when_time_present_but_unmarked():
    from d_brain.services.tmux_parse import unmarked_reset_banner

    text = "  weekly limit reached · resets 9am\n❯\n"
    assert unmarked_reset_banner(text) == "weekly limit reached · resets 9am"


def test_unmarked_reset_banner_none_when_time_is_parseable():
    from d_brain.services.tmux_parse import unmarked_reset_banner

    text = "  weekly limit reached · resets 9am (UTC)\n❯\n"
    assert unmarked_reset_banner(text) is None


def test_unmarked_reset_banner_none_without_a_limit_banner():
    from d_brain.services.tmux_parse import unmarked_reset_banner

    assert unmarked_reset_banner(READY_CAPTURE) is None


def test_unmarked_reset_banner_none_when_banner_has_no_time():
    from d_brain.services.tmux_parse import unmarked_reset_banner

    assert unmarked_reset_banner("  Claude usage limit reached.\n❯\n") is None


# ── reset_epoch: shared anchoring math for watchdog + cron_runner ────────
#
# agent-infra: the cron pane's own recovery (cron_runner
# .CronRunner._limit_recovery) needs the exact same "bare wall-clock time →
# instant" anchoring watchdog._reset_deadline already had, so this helper is
# the one implementation both callers delegate to.


def _epoch(day: str, hhmm: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(f"{day}T{hhmm}:00+00:00").timestamp()


def test_reset_epoch_real_incident_banner_has_no_utc_marker_returns_none():
    """The REAL banner that caused the 2026-08-28 6-day cron deadlock
: "resets 4am (Europe/Amsterdam)" — a
    timezone name, not the "(UTC)"/"UTC" marker parse_reset_time requires
    on the same line. This is not a hypothetical fixture; it's the literal
    text captured live off the parked pane. reset_epoch must return None
    here (so the caller's LIMIT_MAX_WAIT fallback is what actually saves
    it — NOT a reset_at computed from this unreadable time)."""
    banner = "You've hit your weekly limit · resets 4am (Europe/Amsterdam)"
    text = f"  {banner}\n❯\n"
    seen_at = _epoch("2026-08-28", "23:07")
    assert reset_epoch(text, seen_at=seen_at) is None


def test_reset_epoch_utc_banner_rolls_forward_across_midnight():
    """Seen at 22:00 UTC, banner says "resets 12am (UTC)" (midnight, already
    passed for today relative to 22:00 only in the sense that 00:00 < 22:00)
    — the reset must anchor to the NEXT occurrence of that time-of-day, i.e.
    00:00 the following day, mirroring watchdog._reset_deadline's existing
    rollover semantics (see test_limit_nudges_once_the_reset_time_has_passed
    in test_watchdog.py)."""
    banner = "  You've hit your session limit · resets 12am (UTC)\n❯\n"
    seen_at = _epoch("2026-08-20", "22:00")
    assert reset_epoch(banner, seen_at=seen_at) == _epoch("2026-08-21", "00:00")


def test_reset_epoch_utc_banner_same_day_when_time_still_ahead():
    """Seen at 10:00 UTC, banner says "resets at 15:30 (UTC)" — later the
    same day, no rollover needed."""
    banner = "  session limit reached · resets at 15:30 (UTC)\n❯\n"
    seen_at = _epoch("2026-08-20", "10:00")
    assert reset_epoch(banner, seen_at=seen_at) == _epoch("2026-08-20", "15:30")


def test_reset_epoch_none_without_a_limit_banner():
    assert reset_epoch(READY_CAPTURE, seen_at=_epoch("2026-08-20", "10:00")) is None


# ── find_unhandled_replies ───────────────────────────────────────────────


def _pair(rid: str, body: str) -> str:
    return f"<<<R:{rid}>>>\n{body}\n<<<E:{rid}>>>\n"


def test_find_unhandled_returns_every_pending_pair_oldest_first():
    """The loss case: a background reply is superseded by a later turn
    before the poller runs. Only-the-latest lookup dropped it forever."""
    text = _pair("aaa00001", "subagent result") + _pair("bbb00002", "live answer")
    assert find_unhandled_replies(text, set()) == [
        ("aaa00001", "subagent result"),
        ("bbb00002", "live answer"),
    ]


def test_find_unhandled_skips_already_delivered_rids():
    text = _pair("aaa00001", "old") + _pair("bbb00002", "new")
    assert find_unhandled_replies(text, {"aaa00001"}) == [("bbb00002", "new")]


def test_find_unhandled_skips_an_older_pair_repainted_below_a_newer_one():
    """The duplication case: the TUI repaints its transcript, so an
    already-delivered pair can end up being the LAST one on the pane."""
    text = _pair("bbb00002", "new") + _pair("aaa00001", "old, already sent")
    assert find_unhandled_replies(text, {"aaa00001", "bbb00002"}) == []


def test_reply_rids_lists_every_complete_pair():
    text = _pair("aaa00001", "one") + _pair("bbb00002", "two")
    assert reply_rids(text) == {"aaa00001", "bbb00002"}


# ── find_pending_replies: the append-order watermark ─────────────────────


def test_pending_skips_an_older_pair_that_only_now_became_visible():
    """Observed live 2026-08-20: widening the pane made the TUI reflow its
    transcript, so replies from hours earlier fit into the capture window and
    looked unseen. Anything ABOVE a known-delivered pair is older than it."""
    text = (
        _pair("old00001", "Ответ часовой давности")
        + _pair("bbb00002", "уже доставлен")
        + _pair("ccc00003", "новый, не доставлен")
    )
    assert find_pending_replies(text, {"bbb00002"}) == [
        ("ccc00003", "новый, не доставлен")
    ]


def test_pending_returns_everything_when_no_watermark_is_visible():
    """Nothing known on screen ⇒ no ordering to trust; a genuinely fresh
    pane must still deliver."""
    text = _pair("aaa00001", "one") + _pair("bbb00002", "two")
    assert find_pending_replies(text, set()) == [
        ("aaa00001", "one"),
        ("bbb00002", "two"),
    ]


def test_pending_ignores_a_repainted_older_pair_below_the_watermark():
    text = _pair("bbb00002", "new") + _pair("aaa00001", "old, already sent")
    assert find_pending_replies(text, {"aaa00001", "bbb00002"}) == []


# ── unwrap_soft_breaks ───────────────────────────────────────────────────
#
# capture-pane returns the RENDERED screen, so the terminal's wrap points are
# indistinguishable from real newlines and reached Telegram as a ragged
# column. Confirmed live 2026-08-20: even at 200 columns a 447-char sentence
# came back as three ~195-char lines.

_LONG_A = "Второй мозг нужен для того, чтобы не полагаться на память " + "и" * 140
_LONG_B = "  в потоке ежедневных задач и целей, освобождая внимание."


def test_soft_wrapped_sentence_is_rejoined():
    out = unwrap_soft_breaks(f"{_LONG_A}\n{_LONG_B}")
    assert out.count("\n") == 0
    assert out.endswith("освобождая внимание.")
    assert "памяти" not in out  # joined with a space, not glued


def test_short_lines_are_left_alone():
    """A deliberate short line is not a wrap — never merge it."""
    text = "Готово.\nСохранил в vault.\nЧто дальше?"
    assert unwrap_soft_breaks(text) == text


def test_blank_line_paragraph_break_survives():
    text = f"{_LONG_A}\n\nНовый абзац."
    assert "\n\nНовый абзац." in unwrap_soft_breaks(text)


def test_list_item_after_a_long_line_is_not_merged():
    text = f"{_LONG_A}\n- первый пункт\n- второй пункт"
    out = unwrap_soft_breaks(text)
    assert "\n- первый пункт" in out
    assert "\n- второй пункт" in out


def test_numbered_item_after_a_long_line_is_not_merged():
    text = f"{_LONG_A}\n1. первый\n2. второй"
    assert "\n1. первый" in unwrap_soft_breaks(text)


def test_code_block_breaks_are_untouched():
    """Inside <pre>/``` every newline is meaningful."""
    text = f"<pre>\n{_LONG_A}\n  продолжение внутри кода\n</pre>"
    out = unwrap_soft_breaks(text)
    assert "\n  продолжение внутри кода" in out


def test_extract_reply_delivers_an_unwrapped_paragraph():
    """End to end: the wrap must be gone by the time a reply is delivered."""
    text = f"<<<R:wrap0001>>>\n{_LONG_A}\n{_LONG_B}\n<<<E:wrap0001>>>\n"
    body = extract_reply(text, "wrap0001")
    assert body is not None and "\n" not in body


# ── is_main_turn_active / main_turn_finished (-08-21) ─
#
# ask()'s stall loop never escapes when a background-agent list row is on
# screen — is_working()/is_working_progressing() are (correctly) True
# forever while ANY such row ticks, even once the MAIN turn is long done.
# These predicates answer the narrower question ask() actually needs: is the
# turn IT sent a prompt to still running, excluding background-agent rows.

_FOOTER_LINE = "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"

# The TUI's turn-summary verb is RANDOMIZED — measured 2026-08-22 over 258
# real summary lines in ~/.dbrain/pane.log: Worked 53, Brewed 42, Baked 39,
# Cooked 34, Crunched 33, Churned 31, Cogitated 26. A fixture hardcoding only
# "Worked for" encodes the same idealisation the production bug was hiding
# behind — every test below that depends on the
# turn-summary line is parametrized over all seven measured verbs.
_SUMMARY_VERBS = (
    "Worked",
    "Brewed",
    "Baked",
    "Cooked",
    "Crunched",
    "Churned",
    "Cogitated",
)


def test_is_main_turn_active_true_on_legacy_hint():
    assert is_main_turn_active("  ✻ Working…  (esc to interrupt)\n") is True


def test_is_main_turn_active_true_on_paren_anchored_spinner():
    assert (
        is_main_turn_active("Warping… (2m 33s · ↓ 8.4k tokens · thought for 38s)\n")
        is True
    )


def test_is_main_turn_active_true_on_hour_format_spinner():
    """Round-2 finding: the elapsed-time clause only accounted for
    minutes+seconds, so an hour-scale readout ('(1h 5m 12s · ↓ 9.9k
    tokens)') false-negatived on a genuinely still-live main turn."""
    assert is_main_turn_active("Warping… (1h 5m 12s · ↓ 9.9k tokens)\n") is True


def test_is_main_turn_active_true_while_waiting_on_a_background_agent():
    pane = "✻ Waiting for 1 background agent to finish\n" + _FOOTER_LINE
    assert is_main_turn_active(pane) is True


@pytest.mark.parametrize("verb", _SUMMARY_VERBS)
def test_is_main_turn_active_false_on_worked_for_summary_with_agent_rows(verb):
    """The MAIN turn already returned control here — only background
    bookkeeping remains. Must NOT be mistaken for the main turn still being
    active (that mistake is Defect A: it is exactly why ask()'s stall loop
    never escaped this frame shape)."""
    pane = (
        f"{verb} for 2m 22s · 2 background tasks still running\n"
        "  ◯ general-purpose  Inspecting panes   54s · ↓ 79.7k tokens\n"
        "  ◯ general-purpose  Reading files       12s · ↓ 8.1k tokens\n" + _FOOTER_LINE
    )
    assert is_main_turn_active(pane) is False


def test_is_main_turn_active_false_on_bare_idle_prompt():
    pane = "some transcript above\n❯\n" + _FOOTER_LINE
    assert is_main_turn_active(pane) is False


def test_main_turn_finished_false_on_empty_or_whitespace():
    """Fails closed: an empty/garbled capture must never read as finished."""
    assert main_turn_finished("") is False
    assert main_turn_finished("   \n\t  ") is False


def test_main_turn_finished_false_while_main_turn_active():
    pane = "  ✻ Working…  (esc to interrupt)\n" + _FOOTER_LINE
    assert main_turn_finished(pane) is False


@pytest.mark.parametrize("verb", _SUMMARY_VERBS)
def test_main_turn_finished_true_on_worked_for_line(verb):
    pane = (
        f"{verb} for 2m 22s · 2 background tasks still running\n"
        "  ◯ general-purpose  still going   54s · ↓ 79.7k tokens\n" + _FOOTER_LINE
    )
    assert main_turn_finished(pane) is True


def test_main_turn_finished_true_on_bare_idle_prompt():
    assert main_turn_finished("transcript\n❯\n" + _FOOTER_LINE) is True


def test_main_turn_finished_false_on_pure_absence_without_positive_signal():
    """Absence of activity alone is not enough — a positive finished signal
    (Worked-for line or bare idle prompt) must be present too."""
    pane = "some random garbled text with nothing recognizable"
    assert main_turn_finished(pane) is False


# ── extract_open_reply (salvage extraction) ──────────────

_BOX_RULE = "─" * 20


def test_extract_open_reply_returns_body_for_unterminated_span():
    rid = "open0001"
    pane = f"<<<R:{rid}>>>\nHello, this is the salvaged answer.\n"
    assert extract_open_reply(pane, rid) == "Hello, this is the salvaged answer."


def test_extract_open_reply_none_when_complete_pair_exists():
    """Defers to extract_reply — never shadows the authoritative path."""
    rid = "open0002"
    pane = f"<<<R:{rid}>>>\nDone.\n<<<E:{rid}>>>\n"
    assert extract_open_reply(pane, rid) is None


def test_extract_open_reply_none_when_r_marker_absent():
    assert extract_open_reply("no markers here at all\n❯\n", "open0003") is None


def test_extract_open_reply_stops_at_box_rule():
    rid = "open0004"
    pane = f"<<<R:{rid}>>>\nBody text.\n{_BOX_RULE}\ntrailing junk\n"
    assert extract_open_reply(pane, rid) == "Body text."


@pytest.mark.parametrize("verb", _SUMMARY_VERBS)
def test_extract_open_reply_stops_at_worked_for_line(verb):
    rid = "open0005"
    pane = (
        f"<<<R:{rid}>>>\nBody text.\n"
        f"{verb} for 2m 22s · 2 background tasks still running\n"
        "trailing junk\n"
    )
    assert extract_open_reply(pane, rid) == "Body text."


def test_extract_open_reply_stops_at_footer():
    rid = "open0006"
    pane = f"<<<R:{rid}>>>\nBody text.\n{_FOOTER_LINE}trailing junk\n"
    assert extract_open_reply(pane, rid) == "Body text."


def test_extract_open_reply_stops_at_bare_idle_prompt():
    rid = "open0007"
    pane = f"<<<R:{rid}>>>\nBody text.\n❯\ntrailing junk\n"
    assert extract_open_reply(pane, rid) == "Body text."


def test_extract_open_reply_stops_at_paren_anchored_spinner():
    """Round-2 finding: extract_open_reply used to swallow a live
    paren-anchored main-turn spinner line into the salvage candidate if it
    appeared before the next TUI boundary. Recognizing _MAIN_SPINNER_RE as a
    boundary keeps a live spinner out of a salvaged reply's text."""
    rid = "open0010"
    pane = (
        f"<<<R:{rid}>>>\nBody text.\nWarping… (2m 33s · ↓ 8.4k tokens)\ntrailing junk\n"
    )
    assert extract_open_reply(pane, rid) == "Body text."


@pytest.mark.parametrize("verb", _SUMMARY_VERBS)
def test_extract_open_reply_does_not_stop_at_bare_non_paren_spinner(verb):
    """Deliberately narrower than _WORKING_RE: extract_open_reply's boundary
    only recognizes the well-formed, paren-anchored main spinner
    (_MAIN_SPINNER_RE). A non-parenthesized spinner shape (the CLI shape
    behind the round-2 false-negative in is_main_turn_active) is NOT treated
    as a boundary here — it is caught instead by the separate _WORKING_RE
    conjunct in ask()'s salvage condition (claude_session.py), which is the
    intended defense-in-depth split: this function stays narrow/precise, the
    caller's conjunct is the broad safety net. See
    test_salvage_refuses_when_region_contains_a_working_signature in
    test_claude_session.py for the end-to-end proof."""
    rid = "open0011"
    pane = (
        f"<<<R:{rid}>>>\nBody text.\n"
        "✢ Razzle-dazzling…  44s · ↓1.8k tokens\n"
        f"{verb} for 2m 22s · 1 background task still running\n"
    )
    body = extract_open_reply(pane, rid)
    assert body is not None
    assert "44s · ↓1.8k tokens" in body  # deliberately NOT stopped here


def test_extract_open_reply_applies_soft_break_unwrapping():
    rid = "open0008"
    pane = f"<<<R:{rid}>>>\n{_LONG_A}\n{_LONG_B}\n"
    body = extract_open_reply(pane, rid)
    assert body is not None and "\n" not in body


def test_extract_open_reply_none_when_empty_after_stripping():
    rid = "open0009"
    pane = f"<<<R:{rid}>>>\n   \n{_FOOTER_LINE}"
    assert extract_open_reply(pane, rid) is None


def test_extract_open_reply_ignores_the_marker_instruction_echo():
    """Real, recurring input: every Telegram-relayed prompt ends with this
    exact instruction, so its own inline echo of <<<R:id>>>/<<<E:id>>> (with
    real prose after each marker on the SAME line) must never be mistaken
    for a genuine line-anchored marker."""
    rid = "id"
    echo = (
        "❯ some prompt text\n\n"
        "  When done, wrap your ENTIRE reply between a line containing only "
        f"<<<R:{rid}>>> and a line containing only <<<E:{rid}>>>.\n"
    )
    assert extract_open_reply(echo, rid) is None
    # And once the model's REAL (line-anchored) open span follows the echo,
    # only the real one is picked up — not the echo.
    pane = echo + f"⏺ <<<R:{rid}>>>\nReal salvaged answer.\n"
    assert extract_open_reply(pane, rid) == "Real salvaged answer."


# ── golden fixture (T3): a hand-constructed but
# byte-faithful reconstruction of a real incident pane shape — a long
# (~239-line) reply transcript ending at a turn-summary line with NO closing
# marker, followed by the real 11-line bottom chrome (box rule, agent rows,
# the footer WITH "esc to interrupt" fused into it, and a bare idle input
# line). The reply body itself is synthetic filler (no real conversation
# content was captured or committed here) — only the surrounding chrome
# shapes are load-bearing for this test.


def _load_incident_fixture() -> str:
    return (_FIXTURES_DIR / "pane_salvage_incident.txt").read_text(encoding="utf-8")


def test_golden_incident_fixture_main_turn_finished():
    text = _load_incident_fixture()
    assert is_main_turn_active(text) is False
    assert main_turn_finished(text) is True


def test_golden_incident_fixture_extract_open_reply_stops_before_summary_and_chrome():
    text = _load_incident_fixture()
    body = extract_open_reply(text, "gold0001")
    assert body is not None
    # Ends at the last reply line, not swallowing the turn-summary line, the
    # box rule, the footer, or the idle prompt.
    assert body.rstrip().endswith("season.")
    assert "Cooked for" not in body
    assert "─" * 10 not in body
    assert "bypass permissions on" not in body
    assert "esc to interrupt" not in body
    assert not _WORKING_RE.search(body)


# ── open_reply_rids / has_marker (R2a / R2c, Fable audit) ────────────────


def test_open_reply_rids_includes_unclosed_and_closed():
    text = "<<<R:a1>>>\nopen body\n<<<R:b2>>>\nclosed body\n<<<E:b2>>>\n"
    assert open_reply_rids(text) == {"a1", "b2"}


def test_open_reply_rids_empty_when_no_markers():
    assert open_reply_rids("nothing here\n❯\n") == set()


def test_open_reply_rids_ignores_inline_echo():
    rid = "echoid1"
    echo = f"> reply, wrap between <<<R:{rid}>>> and <<<E:{rid}>>> markers\n"
    assert open_reply_rids(echo) == set()


def test_has_marker_true_for_line_anchored_r():
    rid = "hm0001"
    assert has_marker(f"⏺ <<<R:{rid}>>>\ntext\n", rid, "R") is True


def test_has_marker_false_when_absent():
    assert has_marker("no markers\n", "hm0002", "R") is False


def test_has_marker_ignores_inline_echo():
    rid = "hm0003"
    echo = f"> wrap between <<<R:{rid}>>> and <<<E:{rid}>>> markers\n"
    assert has_marker(echo, rid, "R") is False


def test_has_marker_distinguishes_r_and_e():
    rid = "hm0004"
    text = f"<<<R:{rid}>>>\nbody\n<<<E:{rid}>>>\n"
    assert has_marker(text, rid, "R") is True
    assert has_marker(text, rid, "E") is True
    assert has_marker(text, "someone-else", "R") is False


# ── strip_open_reply_body (R5 / F4, Fable audit) ─────────────────────────


def test_strip_open_reply_body_removes_body_keeps_marker_and_boundary():
    rid = "so0001"
    text = f"<<<R:{rid}>>>\nSecret body text.\n{_BOX_RULE}\nafter\n"
    out = strip_open_reply_body(text, rid)
    assert "Secret body text." not in out
    assert f"<<<R:{rid}>>>" in out
    assert _BOX_RULE in out
    assert "after" in out


def test_strip_open_reply_body_noop_when_r_marker_absent():
    text = "no markers here\n❯\n"
    assert strip_open_reply_body(text, "so0002") == text


def test_strip_open_reply_body_noop_when_pair_is_complete():
    """A complete pair is handled by strip_reply_bodies already — this
    function only ever touches an UNTERMINATED span."""
    rid = "so0003"
    text = f"<<<R:{rid}>>>\nDone.\n<<<E:{rid}>>>\n"
    assert strip_open_reply_body(text, rid) == text


def test_strip_open_reply_body_removes_rate_limit_looking_text_in_open_span():
    """F4 (Fable audit): the model's OWN in-progress prose quoting rate-limit
    text must not poison classify_state() once the open span is stripped —
    the specific bug this exists to close."""
    rid = "so0004"
    text = (
        f"<<<R:{rid}>>>\n"
        "Discussing the bug: usage limit reached · resets 12am (UTC)\n"
        f"{_FOOTER_LINE}"
    )
    stripped = strip_open_reply_body(text, rid)
    assert classify_state(text) == PaneState.RATE_LIMITED  # before the fix
    assert classify_state(stripped) != PaneState.RATE_LIMITED  # after


def test_strip_open_reply_body_keeps_a_real_banner_positioned_after_the_span():
    """A REAL rate-limit banner appearing AFTER the open span (outside the
    region this strips) must stay fully visible to classify_state()."""
    rid = "so0005"
    text = (
        f"<<<R:{rid}>>>\n"
        "Some in-progress reply text, nothing about limits here.\n"
        f"{_BOX_RULE}\n"
        "Claude usage limit reached. Your limit will reset at 3pm (UTC)\n❯\n"
    )
    stripped = strip_open_reply_body(text, rid)
    assert classify_state(stripped) == PaneState.RATE_LIMITED



# ── clean-server rehearsal: auth failure during a turn ───────────────────

_TURN_ECHO = (
    "❯ Привет\n  When done, wrap your ENTIRE reply between a line containing only "
    "<<<R:ab12>>> and a line containing only <<<E:ab12>>>. The reply is delivered "
    "ONLY after the <<<E:ab12>>> line — end with\n  it.\n"
)


def test_turn_auth_error_after_the_current_prompt():
    from d_brain.services.tmux_parse import turn_auth_error

    pane = (
        _TURN_ECHO
        + "  ⎿  Invalid API key · Fix external API key\n\n✻ Worked for 3m\n❯ \n"
    )
    assert turn_auth_error(pane, "ab12")
    oauth = _TURN_ECHO + "  ⎿  API Error: 401 · Please run /login\n\n❯ \n"
    assert turn_auth_error(oauth, "ab12")
    expired = (
        _TURN_ECHO
        + "  ⎿  API Error: 401 OAuth token has expired · Please run /login\n❯ \n"
    )
    assert turn_auth_error(expired, "ab12")


def test_turn_auth_error_ignores_model_text_and_other_turns():
    from d_brain.services.tmux_parse import turn_auth_error

    model_says = _TURN_ECHO + "● Если видите «Invalid API key», выполните /login\n"
    assert not turn_auth_error(model_says, "ab12")
    # Tool results share the "⎿" gutter (second review, M1).
    tool = (
        _TURN_ECHO
        + "● Bash(curl -s https://api.anthropic.com/v1/messages)\n"
        + '  ⎿  {"type":"error","error":{"type":"authentication_error"}}\n'
        + "● Search(pattern: \"please run /login\")\n"
        + "  ⎿  API Error: 401 · Please run /login\n"
    )
    assert not turn_auth_error(tool, "ab12")
    old_turn = "  ⎿  Invalid API key\n" + _TURN_ECHO + "● ответ\n"
    assert not turn_auth_error(old_turn, "ab12")
    assert not turn_auth_error(_TURN_ECHO + "  ⎿  Invalid API key\n", "other")



def test_turn_auth_error_after_a_tool_call_in_the_same_turn():
    # Third review: the refresh can fail after the model already ran a tool.
    from d_brain.services.tmux_parse import turn_auth_error

    pane = (
        _TURN_ECHO
        + "● Bash(ls)\n  ⎿  notes.md\n     goals\n\n"
        + "  ⎿  API Error: 401 OAuth token has expired · Please run /login\n\n"
        + "✻ Worked for 12s\n" + "─" * 40 + "\n❯ \n" + "─" * 40 + "\n"
        + "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    assert turn_auth_error(pane, "ab12")
    # ... but not when that last "⎿" line is the tool's own result
    tool_last = (
        _TURN_ECHO + "● Bash(cat log)\n  ⎿  API Error: 401 · Please run /login\n❯ \n"
    )
    assert not turn_auth_error(tool_last, "ab12")
    # ... nor when the model keeps writing after it
    continues = pane.replace("✻ Worked for 12s", "● Продолжаю работу")
    assert not turn_auth_error(continues, "ab12")



_LONG_401 = (
    '  ⎿  API Error: 401 {"type":"error","error":{"type":"authentication_error",'
    '"message":"OAuth token has expired. Please obtain a new token or refresh your '
    'existing token."},"request_id":"req_011CTestTestTestTest"} · Please run\n'
    "     /login\n"
)
_FOOTER = (
    "\n✻ Cooked for 41s\n\n" + "─" * 60 + "\n❯ \n" + "─" * 60 + "\n"
    "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    "  Context left until auto-compact: 12%\n"
)


def test_turn_auth_error_real_wrapped_401_and_footers():
    # Fourth review: the real message wraps at 200 columns; footers vary.
    from d_brain.services.tmux_parse import turn_auth_error

    assert turn_auth_error(_TURN_ECHO + _LONG_401 + _FOOTER, "ab12")
    after_tool = (
        _TURN_ECHO + "● Bash(ls)\n  ⎿  notes.md\n     goals\n\n" + _LONG_401 + _FOOTER
    )
    assert turn_auth_error(after_tool, "ab12")


def test_turn_auth_error_not_under_wrapped_or_nested_tool_headers():
    from d_brain.services.tmux_parse import turn_auth_error

    wrapped_header = (
        _TURN_ECHO
        + "● Bash(python x.py --endpoint https://api.example.com/v1/messages \\\n"
        + "      --retries 3)\n  ⎿  Invalid API key\n" + _FOOTER
    )
    assert not turn_auth_error(wrapped_header, "ab12")
    nested = (
        _TURN_ECHO
        + "● Task(check the API)\n  ⎿  Bash(curl …)\n  ⎿  Invalid API key\n"
        + _FOOTER
    )
    assert not turn_auth_error(nested, "ab12")
    model_after = _TURN_ECHO + _LONG_401 + "\n● Продолжаю\n" + _FOOTER
    assert not turn_auth_error(model_after, "ab12")


# ── foreign view (agent-infra, 2026-09-19) ─────────────────
#
# Frames transcribed from ~/.dbrain/pane.log: the main pane switched to the
# background task `night-second-brain` via the "← N agents" view. It still
# classifies READY, which is why nothing noticed; only the label on the
# prompt box's top border tells it apart from the bot's own conversation.

_WIDE_RULE = "─" * 200
_AGENTS_FOOTER = (
    " ⏵⏵ bypass permissions on (shift+tab to cycle) · ← 3 agents · "
    "gh auth login for PR status · 1 memory\n"
)
_FOREIGN_VIEW = (
    " ✻ Baked for 1m 12s · done 5:21 PM\n"
    f" {'─' * 180} night-second-brain ─\n"
    " ❯ \n"
    f" {_WIDE_RULE}\n" + _AGENTS_FOOTER
)
_MAIN_VIEW = (
    " ✻ Baked for 1m 12s · done 5:21 PM\n"
    f" {_WIDE_RULE}\n"
    " ❯ \n"
    f" {_WIDE_RULE}\n" + _AGENTS_FOOTER
)


def test_foreign_view_label_detects_background_task_view():
    assert classify_state(_FOREIGN_VIEW) == PaneState.READY  # the trap
    assert foreign_view_label(_FOREIGN_VIEW) == "night-second-brain"


def test_foreign_view_label_multiword_and_cyrillic():
    frame = _FOREIGN_VIEW.replace(
        "night-second-brain", "Организация мыслей и восстановление спокойствия"
    )
    assert (
        foreign_view_label(frame) == "Организация мыслей и восстановление спокойствия"
    )


def test_foreign_view_label_detects_typed_input_in_foreign_view():
    frame = _FOREIGN_VIEW.replace(
        " ❯ \n", " ❯ стрелка не сработала, перезапусти основную сессию бота\n"
    )
    assert foreign_view_label(frame) == "night-second-brain"


def test_main_view_has_no_foreign_label():
    assert foreign_view_label(_MAIN_VIEW) is None


def test_new_message_divider_is_not_a_view_label():
    """In-transcript divider: rules on both sides, never above the input."""
    frame = (
        f" {'─' * 90} 1 new message {'─' * 90}\n"
        " ● Football done (50/50) — 2 of 11 in wave 1.\n" + _MAIN_VIEW
    )
    assert foreign_view_label(frame) is None


def test_foreign_label_in_scrollback_only_does_not_count():
    """The pane was switched back to the main conversation: the old
    labelled border scrolled up out of the chrome region."""
    frame = _FOREIGN_VIEW + "\n".join(f" line {i}" for i in range(40)) + "\n"
    frame += _MAIN_VIEW
    assert foreign_view_label(frame) is None


def test_model_text_shaped_like_a_label_is_not_a_view_label():
    """Review finding: the model's own "─── Итог ─" heading directly above a
    quoted "❯ …" line looks exactly like a labelled input box. Only the
    border above the BOTTOM-MOST ❯ line (the real input) may count."""
    frame = (
        " ● Разбор.\n"
        f" {'─' * 40} Итог ─\n"
        " ❯ цитата пользовательского промпта\n"
        "   ещё строка ответа\n" + _MAIN_VIEW
    )
    assert foreign_view_label(frame) is None


def test_rename_label_is_read_but_not_judged_here():
    """Live frame, Claude Code 2.1.278, after `/rename probe-main-name` in
    the MAIN conversation: the very same labelled border. The parser only
    reads the label; ClaudeSession decides whether it is foreign."""
    frame = (
        " ❯ /rename probe-main-name\n"
        "   ⎿  Session renamed to: probe-main-name\n"
        f" {'─' * 180} probe-main-name ─\n"
        " ❯ \n"
        f" {_WIDE_RULE}\n"
        "   ⏵⏵ bypass permissions on (shift+tab to cycle) · ← 4 agents\n"
    )
    assert foreign_view_label(frame) == "probe-main-name"


# Live frame, Claude Code 2.1.278: the "← N agents" list itself.
_AGENTS_LIST_VIEW = (
    "Needs input\n"
    " ✻ current session                                  send a prompt to start\n"
    " ✻ night-second-brain                               send test message\n"
    "Completed\n"
    " ∙ night-independent-qa                             independent QA review\n"
    f"{_WIDE_RULE}\n"
    "❯ describe a task for a new session\n"
    f"{_WIDE_RULE}\n"
    "  ⏵⏵ bypass permissions · enter to open · space to reply · ctrl+x to "
    "delete · ? for shortcuts\n"
)


def test_agents_list_view_is_recognised():
    assert is_agents_list_view(_AGENTS_LIST_VIEW)
    # Unlabelled input box: why the list needs a check of its own.
    assert foreign_view_label(_AGENTS_LIST_VIEW) is None


def test_agents_list_view_not_seen_in_session_views():
    assert not is_agents_list_view(_MAIN_VIEW)
    assert not is_agents_list_view(_FOREIGN_VIEW)


def test_list_footer_hint_counts_only_in_the_last_lines():
    """ "ctrl+x to delete" is the list's footer. The same words in the model's
    text higher up (still inside the chrome window) must not turn the bot's
    own conversation into "the agents list"."""
    quoted = (
        " ● In the agents list, press ctrl+x to delete a finished session.\n"
        "   Anything else?\n" + _MAIN_VIEW
    )
    assert not is_agents_list_view(quoted)
    footer_only = (
        f"{_WIDE_RULE}\n❯ \n{_WIDE_RULE}\n  ctrl+x to delete · ? for shortcuts\n"
    )
    assert is_agents_list_view(footer_only)


def test_agents_list_with_a_draft_is_still_the_list():
    """Live, 2.1.278: text typed into the list's input replaces the
    placeholder, and the footer becomes "enter to create · esc to clear"."""
    frame = (
        " ∙ night-independent-qa                             independent QA\n"
        f"{_WIDE_RULE}\n"
        "❯ draft in list\n"
        f"{_WIDE_RULE}\n"
        "  enter to create · esc to clear\n"
    )
    assert is_agents_list_view(frame)


# ── input box (lost Enter, 2026-09-19) ──────────────────────────────────────


def test_input_box_text_reads_a_wrapped_unsent_paste():
    # Live shape (2.1.278): 31-line paste still in the box after Enter.
    frame = (
        "● earlier\n"
        f"{_WIDE_RULE}\n"
        "❯ filler line\n"
        "  filler line\n"
        "  end <<<E:abc123>>>\n"
        f"{_WIDE_RULE}\n"
        "  ⏸ manual mode on\n"
    )
    assert input_box_text(frame) == "filler line\nfiller line\nend <<<E:abc123>>>"


def test_input_box_text_collapsed_paste_and_empty_box():
    box = f"{_WIDE_RULE}\n❯ [Pasted text #2 +200 lines]\n{_WIDE_RULE}\n  paste again\n"
    assert input_box_text(box) == "[Pasted text #2 +200 lines]"
    assert input_box_text(_MAIN_VIEW) == ""
    assert input_box_text(_FOREIGN_VIEW) == ""  # labelled top rule is fine


def test_input_box_text_ignores_the_unboxed_transcript_echo():
    # The echo of a SENT prompt also starts with ❯ and carries the marker,
    # but has no rule directly above it; the empty box below is what counts.
    frame = (
        "❯ ping <<<R:abc>>> and <<<E:abc>>>\n"
        "✻ Working… (esc to interrupt)\n"
        f"{_WIDE_RULE}\n❯ \n{_WIDE_RULE}\n"
    )
    assert input_box_text(frame) == ""
    echo_only = "● answer\n❯ ping <<<E:abc>>>\n  more\n"
    assert input_box_text(echo_only) is None


def test_input_box_text_keeps_an_emptied_last_row():
    frame = f"{_WIDE_RULE}\n❯ human line one\n  \n{_WIDE_RULE}\n"
    assert input_box_text(frame) == "human line one\n"


def test_input_box_text_needs_the_closing_rule():
    assert input_box_text(f"{_WIDE_RULE}\n❯ half a box <<<E:abc>>>\n") is None


# ── two-column frames (2026-09-20 incident) ─────────────────────────────
#
# A pane rendered in two columns (transcript left, a file diff right) puts
# text AFTER the `<<<E:id>>>` marker on its line. The marker regexes require
# the marker at END of line — that is the ONLY thing separating a real answer
# from the prompt echo — so every reply of the second instance parsed as
# region=None and nothing reached its user. The rule is NOT relaxed here: the
# right column is cut off and the same strict rule re-applied.

_TWO_COL_RID = "dfca5573"


def _two_column_frame() -> str:
    return (_FIXTURES_DIR / "pane_two_column.txt").read_text(encoding="utf-8")


def test_two_column_frame_still_finds_the_reply():
    """The regression this whole change exists for."""
    frame = _two_column_frame()
    # Precondition: the end marker really is NOT at the end of its line — the
    # fixture must reproduce the broken shape, not an already-clean frame.
    # The LAST one — the first is the prompt echo's own mention of it.
    marker_line = [ln for ln in frame.splitlines() if f"<<<E:{_TWO_COL_RID}>>>" in ln][
        -1
    ]
    assert not marker_line.rstrip().endswith(">>>")
    assert "+ЬКО" in marker_line  # the right column's diff text

    body = extract_reply(frame, _TWO_COL_RID)
    assert body is not None
    assert "три отдельные правки" in body
    assert is_complete(frame, _TWO_COL_RID)
    assert has_marker(frame, _TWO_COL_RID, "R")
    assert has_marker(frame, _TWO_COL_RID, "E")
    assert reply_rids(frame) == {_TWO_COL_RID}
    assert [r for r, _ in find_unhandled_replies(frame, set())] == [_TWO_COL_RID]
    assert find_latest_reply(frame) == (_TWO_COL_RID, body)


def test_two_column_reply_body_carries_no_right_column_text():
    """Cutting every line at ONE fixed column leaks the first characters of
    any cell that starts further left — the right column's own cells are
    indented differently. Measured; it corrupted the body. Each line is cut
    at its own gutter instead, and this is the regression test for it."""
    body = extract_reply(_two_column_frame(), _TWO_COL_RID)
    for leaked in ("+ЬКО", "+Полив", "Грунт", "+для растений"):
        assert leaked not in body
    assert "10" not in body and "12" not in body  # stray diff line numbers


def test_two_column_frame_still_rejects_the_prompt_echo():
    """The echo lives INSIDE the left column and keeps its own trailing text
    after the marker, so cutting the right column must not promote it to a
    deliverable reply. Without this the 'marker at end of line' rule would be
    relaxed in effect, and every prompt would answer itself."""
    lines = _two_column_frame().split("\n")
    # Rows 10..16 are the model's real answer; drop them, keep the echo.
    echo_only = "\n".join(lines[:10] + lines[17:])
    assert f"<<<R:{_TWO_COL_RID}>>>" in echo_only  # the echo is still there
    assert extract_reply(echo_only, _TWO_COL_RID) is None
    assert extract_open_reply(echo_only, _TWO_COL_RID) is None
    assert not is_complete(echo_only, _TWO_COL_RID)
    assert find_unhandled_replies(echo_only, set()) == []
    assert find_latest_reply(echo_only) is None
    assert open_reply_rids(echo_only) == set()


def test_two_column_open_span_is_salvaged_and_strippable():
    """`region=None` was logged by the SALVAGE path: with text after it the
    OPEN marker was not line-anchored either, so an unclosed answer had no
    delivery path at all."""
    lines = _two_column_frame().split("\n")
    open_only = "\n".join(lines[:15] + lines[16:])  # drop the E marker row
    assert extract_reply(open_only, _TWO_COL_RID) is None
    body = extract_open_reply(open_only, _TWO_COL_RID)
    assert body is not None and "три отдельные правки" in body
    assert open_reply_rids(open_only) == {_TWO_COL_RID}

    # …and that body must also be removable from the RAW frame, or the prose
    # it delivers sits in the chrome window and can fake a RATE_LIMITED.
    stripped = strip_open_reply_body(open_only, _TWO_COL_RID)
    assert "три отдельные правки" not in stripped
    assert f"<<<R:{_TWO_COL_RID}>>>" in stripped  # the marker row survives
    assert "bypass permissions on" in stripped  # chrome below it survives


def test_single_column_frames_are_left_exactly_as_they_are():
    """The fallback must be invisible on a normal pane: no column detected,
    so every marker function sees the raw capture, byte for byte."""
    rid = "abcd1234"
    normal = (
        f"❯ reply and wrap it in <<<R:{rid}>>> and <<<E:{rid}>>> markers\n"
        "✻ Working… (12s · ↓ 300 tokens)\n"
        f"⏺ <<<R:{rid}>>>\n"
        "  Готово, файл обновлён.\n"
        f"  <<<E:{rid}>>>\n"
        f"{_WIDE_RULE}\n❯ \n{_WIDE_RULE}\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    assert strip_right_column(normal) == normal
    assert extract_reply(normal, rid) == "Готово, файл обновлён."


def test_an_indented_block_is_not_mistaken_for_a_column():
    """A diff body / code block is indented but has NOTHING to its left, and
    that is exactly what separates an indent from a real second column.
    Cutting it would delete the block."""
    diff = (
        "⏺ Update(sample.md)\n"
        "  ⎿  Added 3 lines\n"
        "      1  # Грунт\n"
        "      2\n"
        "      3  Мы используем европейский минеральный грунт\n"
        "      4 +для растений в интерьере\n"
        "      5 +и для оранжерей\n"
        "      6 +и для зимних садов\n"
    )
    assert strip_right_column(diff) == diff


def test_real_single_column_captures_detect_no_column():
    """Guard on the detector's precision — the property the whole fallback
    rests on, because a false positive TRUNCATES a reply instead of dropping
    it. Checked against the real captures this suite already carries."""
    for frame in (
        READY_CAPTURE,
        READY_NO_BYPASS_CAPTURE,
        (_FIXTURES_DIR / "pane_salvage_incident.txt").read_text(encoding="utf-8"),
    ):
        assert strip_right_column(frame) == frame


# ── main_area_working (the salvage safety net) ─────────────────────────────

_MAW_BOX = "─" * 80
_MAW_FOOTER = (
    "  ⏵⏵ bypass permissions on · 2 background tasks · esc to interrupt · "
    "← for agents · ↓ to manage\n"
)


def test_main_area_working_sees_the_non_paren_spinner_above_the_box():
    """The shape is_main_turn_active() misses — this is the whole point."""
    pane = (
        "⏺ some reply text\n"
        "✢ Razzle-dazzling…  44s · ↓1.8k tokens\n"
        f"{_MAW_BOX}\n❯\n{_MAW_BOX}\n{_MAW_FOOTER}"
    )
    assert is_main_turn_active(pane) is False
    assert main_area_working(pane) is True


def test_main_area_working_ignores_background_agent_rows_below_the_box():
    """A listed background task must never read as "the main turn is still
    writing" — that is Defect A/B, and the golden incident fixture has
    exactly this shape."""
    pane = (
        "⏺ some reply text\n"
        "Worked for 3m 41s · 1 background task still running\n"
        f"{_MAW_BOX}\n❯\n{_MAW_BOX}\n"
        "  hello | Opus 4.8 (1M context) | ~/p\n"
        f"{_MAW_FOOTER}"
        "  ● main\n"
        "  ◯ general-purpose  Reviewing salvage fix diff   3m 14s · ↓ 42.7k tokens\n"
    )
    assert main_area_working(pane) is False


def test_main_area_working_ignores_agent_rows_drawn_above_the_footer():
    """Same rows, the other measured layout (above the footer, still below
    the prompt box) — the cut is the box, not the footer."""
    pane = (
        "⏺ some reply text\n"
        "Worked for 2m 22s · 1 background task still running\n"
        f"{_MAW_BOX}\n❯\n{_MAW_BOX}\n"
        "  ● main\n"
        "  ◯ general-purpose  still going   12s · ↓ 12k tokens\n"
        f"{_MAW_FOOTER}"
    )
    assert main_area_working(pane) is False


def test_main_area_working_ignores_the_footers_own_esc_to_interrupt():
    """99.6% of real "esc to interrupt" occurrences are the persistent
    footer with the main turn long finished."""
    pane = f"⏺ some reply text\n{_MAW_BOX}\n❯\n{_MAW_BOX}\n{_MAW_FOOTER}"
    assert main_area_working(pane) is False


def test_main_area_working_sees_a_background_agent_wait_above_the_box():
    pane = (
        "⏺ some reply text\n"
        "✻ Waiting for 1 background agent to finish\n"
        f"{_MAW_BOX}\n❯\n{_MAW_BOX}\n{_MAW_FOOTER}"
    )
    assert main_area_working(pane) is True


def test_main_area_working_finds_the_box_from_the_bottom_not_the_top():
    """A markdown rule in the model's own reply renders as the same box-rule
    shape, and the reply is IN this window (only closed pairs are stripped).
    Cutting at the first one from the top hid the live spinner below it —
    the second-round review finding."""
    pane = (
        "⏺ some reply text\n"
        "Here is the summary:\n"
        f"{_MAW_BOX}\n"
        "More to check.\n"
        "✢ Razzle-dazzling…  44s · ↓1.8k tokens\n"
        f"{_MAW_BOX}\n❯\n{_MAW_BOX}\n{_MAW_FOOTER}"
    )
    assert main_area_working(pane) is True


def test_main_area_working_survives_a_quoted_prompt_line_in_the_reply():
    """Same trap via `_IDLE_BARE_RE`: the model quoting a bare ❯ line."""
    pane = (
        "⏺ some reply text\n"
        "❯\n"
        "✢ Razzle-dazzling…  44s · ↓1.8k tokens\n"
        f"{_MAW_BOX}\n❯\n{_MAW_BOX}\n{_MAW_FOOTER}"
    )
    assert main_area_working(pane) is True
