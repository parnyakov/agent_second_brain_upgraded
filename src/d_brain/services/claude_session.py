"""Drive a persistent INTERACTIVE Claude Code session inside a tmux pane.

This is the core of the post-2026-06-15 design: instead of spawning
`claude -p` per request (which moves to a paid Agent SDK credit), we keep
one long-lived interactive `claude` alive in tmux and "type" prompts into
it. Interactive usage stays on the subscription.

All access to the pane goes through a single cross-process file lock
(`pane.lock`) so the bot, the daily pipeline and the watchdog never talk to
the pane at once. The watchdog recovers a wedged session via
``force_recover()`` (non-blocking lock), and ``ask()`` self-detects a stall
(no visible turn — the working spinner gone without completion) so it
releases the lock quickly instead of holding it for the full timeout.
Silence is NOT a hang signal: a long quiet task still shows the spinner.

Pure text parsing lives in tmux_parse; this module only orchestrates tmux
and timing, so it is tested with a fake runner + injected clock/sleep/rid.

NOTE: runtime_dir must be on a LOCAL filesystem — fcntl.flock is unreliable
on NFS/9p and would silently degrade to no serialization.
"""

import fcntl
import json
import logging
import os
import re
import shlex
import subprocess
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from d_brain.services import long_run
from d_brain.services.tmux_parse import (
    PaneState,
    _chrome,
    classify_state,
    extract_open_reply,
    extract_reply,
    find_pending_replies,
    foreign_view_label,
    has_survey_prompt,
    input_box_text,
    is_agents_list_view,
    is_complete,
    is_idle,
    is_main_turn_active,
    is_working,
    is_working_progressing,
    main_area_working,
    main_turn_finished,
    open_reply_rids,
    reply_rids,
    strip_chrome,
    strip_open_reply_body,
    turn_auth_error,
)
from d_brain.services.transcript import (
    ReplyTail,
    latest_reply,
    transcript_dir,
    transcript_path,
)

logger = logging.getLogger(__name__)

Runner = Callable[..., subprocess.CompletedProcess]

# Backstop against a runaway turn, NOT the hang detector — that is
# DEFAULT_STALL_TIMEOUT below, which watches for lack of *progress*. Raised
# 1200 → 3600 on 2026-08-20: three real turns (10:05, 10:26, 11:16 UTC) hit
# the 20 min ceiling while still legitimately progressing through long
# tool-dense work, and lost their reply. An hour is past the point where a
# waiting Telegram user still wants the answer, so it stays a ceiling.
#
# COUPLED to delivery_guard.DEFAULT_WINDOW (= 3x max(this,
# DEFAULT_STALL_TIMEOUT)): that guard only restarts the bot on failures
# CLOSE together, so if this ceiling grows without the window growing too,
# a handful of genuine failures can span longer than the window and the
# auto-restart safety net silently stops firing. delivery_guard derives its
# window from these two constants, so raising either one here moves it
# automatically — but re-check DEFAULT_RESTART_COOLDOWN there still makes
# sense against the new window.
DEFAULT_TIMEOUT = 3600
# No liveness signal for this long ⇒ wedged. Raised 180 → 900 on 2026-08-20:
# four logged incidents were live 6-9 minute turns (dozens of chained tool
# calls, background subagents) reported to the user as "❌ Ошибка сессии".
# 180s only ever bought a faster error message — the hard DEFAULT_TIMEOUT
# still bounds the turn, and a genuinely dead session is the watchdog's job.
# Same delivery_guard.DEFAULT_WINDOW coupling as DEFAULT_TIMEOUT above.
#
# THIS RAISE IS ONLY SAFE TOGETHER WITH is_working_progressing() (tmux_parse):
# a widened _WORKING_RE plus a long stall_timeout means a STATIC "working"
# signature (a dead subagent, a process wedged in D-state, a stuck heredoc)
# would otherwise be trusted as liveness for the full 900s before either
# self- or watchdog-recovery — a real class of hang, not hypothetical (2026-08
# postmortem debt #3). is_working_progressing() requires the PROGRESS
# signatures to actually change frame-to-frame, so 900s here means "15
# minutes with no change AND no pane.log growth", not "15 minutes of a stale
# spinner". Do not raise this further without re-verifying that guard still
# holds, and do not narrow is_working_progressing back to plain is_working
# while this stays at 900 (see ask()'s stall loop and the pre-send busy-wait
# below, both of which depend on it).
DEFAULT_STALL_TIMEOUT = 900
# How long ask() will wait — pane.lock AND the process-wide ask-lock held —
# for a LEFTOVER previous turn to clear before giving up and refusing to
# type over it. Deliberately NOT DEFAULT_STALL_TIMEOUT: that budget sizes
# ask()'s OWN stall detection against the widened liveness window above, but
# reusing all 900s of it here for someone ELSE's leftover turn meant a
# single stuck pane could block incoming chat input for up to 15 minutes —
# during which chat.py turns messages away with a "повтори через пару
# минут" promise it does not keep, AND the watchdog's force_recover() can't
# take the lock either, since it only ever tries non-blocking (round-2
# finding, 2026-08-20). Sized to watchdog.DEFAULT_STALL_THRESHOLD (300s)
# instead, so this gives up around the same time the watchdog would
# otherwise notice the pane is wedged and step in.
DEFAULT_BUSY_WAIT_BUDGET = 300
# How long the pane must show a finished, not-working main turn AND the
# transcript must hold a byte-identical body for an unterminated <<<R:id>>>
# span before ask() salvages the reply from that span instead of waiting for
# a closing <<<E:id>>> that may never come (-08-21: the
# model occasionally drops the literal closing marker for an otherwise-
# complete answer). Stability is checked on the reply BODY only, never on the
# whole pane capture — the whole pane changes every second from ticking
# background-agent rows, which would defeat any whole-pane stability check
# entirely. Sized well below
# DEFAULT_STALL_TIMEOUT (900s) so a genuinely missing-marker reply reaches
# the user in ~2 minutes instead of riding the stall/timeout ceiling all the
# way to an hour.
DEFAULT_SALVAGE_STABLE = 120.0
# Hard ceiling for "main turn not active, no completion detected, salvage
# not possible (or not yet stable)" — the honest-timeout fallback for when
# salvage cannot recover a reply (e.g. the <<<R:id>>> marker itself scrolled
# out of the capture window). Sized to match DEFAULT_BUSY_WAIT_BUDGET and
# watchdog.DEFAULT_STALL_THRESHOLD (both 300s): by the time this fires, the
# watchdog would independently be reaching the same conclusion about the
# pane. Deliberately NEVER feeds into delivery_guard.DEFAULT_WINDOW, which
# stays exactly `3 * max(DEFAULT_STALL_TIMEOUT, DEFAULT_TIMEOUT)`.
DEFAULT_NO_MAIN_TURN_CEILING = 300.0
# How long a byte-identical pane (no chrome change AND no pane.log growth)
# may keep being trusted purely on the static "esc to interrupt" hint before
# the liveness predicates stop believing it.
#
# The value is pinned by an INEQUALITY, not by taste — it must sit strictly
# between two hard numbers:
#
#   lower bound  1440s — the longest legitimately-quiet held turn ever
#                measured here (the 24-minute turn from the item-23
#                investigation). Below this the anti-B3 property (never
#                interrupt a real but silent long turn) starts to break.
#   upper bound  DEFAULT_TIMEOUT - DEFAULT_STALL_TIMEOUT = 2700s — ask()'s
#                stall interrupt fires at (window + stall threshold), so at
#                or above this the interrupt cannot fire BEFORE the hard turn
#                timeout and the whole mechanism is dead code on every
#                production call site.
#
# 2700.0 was the original value and it landed exactly ON the upper bound:
# 2700 + DEFAULT_STALL_TIMEOUT(900) == DEFAULT_TIMEOUT(3600) exactly, so on a
# frozen pane the old and new code produced a byte-identical outcome (hard
# timeout at 3600s, no Escape ever sent). The plan that chose it mis-added
# 2700+900 as "~45 min"; it is 60. 1800.0 restores the intended ~45-minute
# detection latency (1800 + 900 = 2700s), keeps a full 900s of headroom under
# DEFAULT_TIMEOUT so the interrupt genuinely fires first, and still leaves a
# 1.25x margin over the 1440s quiet-turn precedent.
# test_ask_frozen_static_pane_stalls_at_production_constants locks the
# inequality in: it runs the real (900, 3600) shape and goes red if either
# end of it is violated again.
STATIC_TRUST_WINDOW = 1800.0
# request_id prefix that marks a turn as maintenance (pipeline, doctor,
# /process) — such turns are never steering targets for chat input.
MAINT_PREFIX = "maint-"
# See ClaudeSession.request_abort.
ABORT_STAMP_NAME = "abort.stamp"
# How stale a long_run.json marker (written by the watchdog) may be before
# ask()'s fast busy-active path stops trusting it and falls back to the full
# busy-wait — see long_run.is_active()'s docstring. Sized well above the
# watchdog's own DEFAULT_TICK (15s) so a couple of missed/slow ticks don't
# spuriously invalidate a genuinely live marker, and well below anything a
# dead watchdog process would leave stale for.
DEFAULT_LONG_RUN_STALE_AFTER = 120.0
# How many confirmation polls ask() spends re-verifying a fresh long_run.json
# marker against the LIVE pane before trusting it (see the busy branch in
# ask()) — evidence-based, not a blind trust of the marker file. Sized so the
# confirmation window is roughly a few seconds at the default poll_interval.
_LONG_RUN_CONFIRM_POLLS = 3
# Blind-review fix (F1, post-implementation round): how RECENT a genuine
# chrome-changed-or-log-grew observation must be, at busy-wait give-up time,
# to count as "this pane is live" for the busy/busy_active split below.
# Deliberately SEPARATE from is_working_progressing() (used elsewhere in this
# same loop for the stall-timeout break): that predicate intentionally
# exempts the static "esc to interrupt" hint forever, because for THAT
# purpose (don't kill a silently-alive turn) a real quiet task holds it
# static for its whole duration. Reusing it here was the original mistake —
# it let a genuinely wedged pane that merely still shows the hint (or that
# changed a few times early on and then froze solid) latch "progress" once
# and never lose it, defeating the B3 regression guard the "busy" status
# exists to preserve. Classification uses ONLY a real, RECENT change:
# _chrome(new) != _chrome(old) or pane.log growth, with no hint exemption,
# and a recency window of _RECENT_PROGRESS_POLLS polls so one early blip
# can't paper over the rest of a wait spent frozen.
_RECENT_PROGRESS_POLLS = 2
# How many consecutive polls must show RATE_LIMITED before ask() trusts it.
# A single sample is not enough (2026-08-20 review, reproduced live): raw
# tool output (Read/grep/cat) is not wrapped in the <<<R>>>/<<<E>>> markers
# strip_reply_bodies() strips, and an OPEN (unterminated) marker span is
# deliberately never stripped either — so one frame where the model's own
# turn happens to be reading or discussing this very bug's banner text can
# land in the chrome window and read as a real limit. A genuine banner is a
# static full-screen state; it survives several 1s-spaced samples, while a
# frame like that is normally gone by the next poll as the turn keeps moving.
_RATE_LIMIT_CONFIRM_POLLS = 3
# DEFAULTS ONLY — the live values are per-instance (pane_width/pane_height
# constructor args, Settings.pane_width / DBRAIN_PANE_WIDTH). Made settable
# 2026-09-20: a wide pane lets the TUI lay the screen out in two columns
# (transcript left, a file diff right), which appends text after the
# `<<<E:id>>>` marker on its line and made every reply of the second
# instance parse as region=None. tmux_parse strips such a column now; a
# narrower pane is the mitigation that needs no parsing to be right.
_PANE_WIDTH = "200"
# Height 50 (not taller): the TUI draws its footer (idle ❯ / bypass line)
# just below content, so on a tall pane the footer lands mid-screen with a
# blank bottom — and chrome-region state detection misses it. At 50 the
# footer sits near the bottom. Long replies are still captured via scrollback.
_PANE_HEIGHT = "50"
# Scrollback lines to capture per poll. Kept modest for CPU/poll cost, not
# because larger counts are broken: re-verified live 2026-08-22 (Fable audit
# R4) on the actual installed tmux (3.2a) that `-S -2000` and `-S -` both
# return full output, with EITHER the (buggy, pre-fix) 2000-line history
# limit or the intended 50000 one — could not reproduce the old claim that
# large/`-S -` counts return empty under any of those conditions. That claim
# is removed rather than kept as unverified folklore; if a future CLI/tmux
# combination reproduces it, re-add a note with the repro. Deep captures
# beyond this default are for the salvage/orphan paths only, if ever needed.
_CAPTURE_SCROLLBACK = "-200"
# Parked sessions (see ClaudeSession._park_foreign_session) kept per brain;
# older ones are killed so they cannot pile up across repeated engine switches.
_MAX_PARKED = 2
# How long each step of the return-to-main keystroke sequence may take to
# show on screen (see ClaudeSession._return_to_main).
_VIEW_SWITCH_TIMEOUT = 5.0
# Pseudo-label for the "← N agents" list view itself.
AGENTS_LIST_VIEW = "← agents list"
# Rounds of C-k + C-u _clear_input_draft may spend on a multi-line draft
# (each round clears about one line, verified live).
_MAX_CLEAR_ROUNDS = 8
# Extra Enters _send_prompt may press when the first one was lost (seen
# live once: five prompts glued together in the input box, never
# sent). Two, not more: a box that still holds the prompt after three Enters
# is not a lost keystroke.
_ENTER_RETRIES = 2
# What an unsent paste leaves in the input box once the CLI collapses it.
_PASTED_TEXT = "[Pasted text"
# Transcript records that carry the session's display name (verified live on
# 2.1.278: `/rename foo` appends custom-title + agent-name records holding
# "foo" to the pinned transcript; the CLI's own auto-title is ai-title).
_TITLE_FIELDS = {
    "custom-title": "customTitle",
    "agent-name": "agentName",
    "ai-title": "aiTitle",
}


def exact_target(session_name: str) -> str:
    """tmux ``-t`` value that matches ONLY ``session_name`` (its active pane).

    See the ``_target`` comment in ``ClaudeSession.__init__`` for why a bare
    name is a bug (prefix match onto ``<name>_cron``) and why the trailing
    ``:`` is required."""
    return f"={session_name}:"


@dataclass
class AskResult:
    """Outcome of a single ask() round."""

    # "ok" | "rate_limited" | "logged_out" | "timeout" | "error" | "busy" |
    # "busy_active" | "reset"
    #
    # "reset": the owner's /reset cut this turn short (see
    # ClaudeSession.request_abort). Not a delivery failure — ask_health
    # scores it as neutral.
    #
    # "busy_active" (agent-infra, 2026-09): the pane is busy
    # with a leftover turn that DEMONSTRABLY MADE PROGRESS across the whole
    # wait — a live, working turn (e.g. mid an unattended agent cascade), not
    # a wedged one. Distinct from plain "busy" (no progress observed —
    # the original B3 case, still treated as a delivery failure) so a
    # healthy-but-busy pane during a long autonomous run stops being
    # misclassified as a delivery outage. See ask_health.py's module
    # docstring for how the two statuses are scored differently.
    status: str
    reply: str | None = None
    detail: str | None = None
    # True iff this reply was recovered from an unterminated <<<R:id>>> span
    # (no closing marker ever appeared) rather than a complete R/E pair.
    # Purely additive/default-valued — see (2026-08-21).
    salvaged: bool = False
    # Elapsed busy-wait time (seconds), set only when status is "busy" or
    # "busy_active" — purely additive/default-valued. Lets callers (e.g.
    # chat_session's user-facing message) mention how long the pane has
    # been busy without re-deriving it themselves.
    busy_seconds: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class ClaudeSession:
    """A single interactive Claude Code session in a named tmux session."""

    def __init__(
        self,
        session_name: str,
        work_dir: Path,
        runtime_dir: Path,
        *,
        mcp_config: Path | None = None,
        system_prompt_file: Path | None = None,
        model: str | None = None,
        claude_bin: str = "claude",
        runner: Runner = subprocess.run,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
        rid_factory: Callable[[], str] | None = None,
        poll_interval: float = 1.0,
        paste_settle: float = 0.3,
        submit_settle: float = 0.5,
        startup_timeout: float = 90.0,
        stall_timeout: float = DEFAULT_STALL_TIMEOUT,
        busy_wait_budget: float = DEFAULT_BUSY_WAIT_BUDGET,
        salvage_stable: float = DEFAULT_SALVAGE_STABLE,
        no_main_turn_ceiling: float = DEFAULT_NO_MAIN_TURN_CEILING,
        tmux_config: Path | None = None,
        long_run_stale_after: float = DEFAULT_LONG_RUN_STALE_AFTER,
        view_switch_timeout: float = _VIEW_SWITCH_TIMEOUT,
        pane_width: int | str | None = None,
        pane_height: int | str | None = None,
    ) -> None:
        self.session_name = session_name
        self.work_dir = Path(work_dir)
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        # The dir holds the full pane transcript and turn state — owner-only,
        # enforced on every start so pre-existing installs get repaired too.
        # A foreign-owned dir (e.g. once created via sudo) must degrade to a
        # loud warning, not a constructor crash-loop across all services.
        try:
            os.chmod(self.runtime_dir, 0o700)
        except OSError as exc:
            logger.warning(
                "could not restrict %s to owner-only: %s", self.runtime_dir, exc
            )
        self.mcp_config = Path(mcp_config) if mcp_config else None
        self.system_prompt_file = (
            Path(system_prompt_file) if system_prompt_file else None
        )
        self.model = model
        self.claude_bin = claude_bin
        self._runner = runner
        self._sleep = sleep_fn
        self._clock = clock_fn
        self._rid_factory = rid_factory or (lambda: uuid.uuid4().hex[:8])
        self._poll_interval = poll_interval
        self._view_switch_timeout = view_switch_timeout
        # Pane geometry, per instance. tmux wants strings; None keeps the
        # module defaults, so an unconfigured install is byte-for-byte
        # unchanged. See _PANE_WIDTH for why this became settable.
        self._pane_width = str(pane_width) if pane_width else _PANE_WIDTH
        self._pane_height = str(pane_height) if pane_height else _PANE_HEIGHT
        self._paste_settle = paste_settle
        self._submit_settle = submit_settle
        self._startup_timeout = startup_timeout
        self._stall_timeout = stall_timeout
        self._busy_wait_budget = busy_wait_budget
        self._salvage_stable = salvage_stable
        self._no_main_turn_ceiling = no_main_turn_ceiling
        # B3: how stale a long_run.json marker
        # may be before the pre-send busy-wait stops trusting it — see
        # DEFAULT_LONG_RUN_STALE_AFTER and long_run.is_active().
        self._long_run_stale_after = long_run_stale_after
        # B4 fix (Fable audit fix round, 2026-08-22): a tmux config file
        # (deploy/tmux.conf) applied via `-f` at the ACTUAL `new-session`
        # invocation — see _ensure_locked(). Needed because the `-g
        # set-option` call right before `new-session` (R4, kept below) only
        # works when a tmux SERVER already exists to hold the global
        # default: on a genuinely cold start (no server running yet, e.g.
        # right after a reboot — exactly the case R4 was fixed for) that
        # `set-option -g` call itself fails ("error connecting to
        # /tmp/tmux-…: No such file or directory") before any server exists
        # to hold the setting, and the `new-session` that follows then gets
        # tmux's own built-in default (2000) instead. Verified live
        # (2026-08-22): `-f <config>` on the SAME invocation that creates
        # the session produces `history_limit=50000` from cold, because it
        # makes tmux read the config file as part of starting the new
        # server, before the session is created. None if the file cannot be
        # found — degrades to the `-g` path alone (still correct on warm
        # starts, which is every restart except the very first boot).
        self.tmux_config = Path(tmux_config) if tmux_config else None
        # Previous capture seen by is_working() — lets that method require
        # the PROGRESS signatures (elapsed-timer, background-agent wait) to
        # show change tick-to-tick, the same guard ask()'s stall loop
        # applies to its own polling (see is_working_progressing).
        self._last_is_working_cap: str | None = None
        # when the pane first went byte-identical (chrome
        # unchanged AND pane.log not growing) as seen by is_working(). None
        # ⇒ the pane changed on the last observation, i.e. nothing frozen to
        # time. Only ever read/written by is_working() — see STATIC_TRUST_WINDOW.
        self._static_frozen_since: float | None = None
        # pane.log size at the previous is_working() call, so that method can
        # tell "frozen chrome because the turn is dead" from "frozen chrome
        # because the streamed reply body is stripped out of chrome while the
        # log keeps growing" (strip_reply_bodies) — the second one is alive.
        self._last_is_working_log_size = 0

        # Address the session's active window/pane by name. A fixed ":0.0"
        # breaks under `base-index 1` (window 0 won't exist) → empty capture.
        #
        # EXACT match (`=name:`), never a bare name: tmux resolves a bare
        # `-t dbrain_X` by PREFIX, so while `dbrain_X` is absent it silently
        # finds `dbrain_X_cron` — has-session says "exists", and every prompt,
        # capture and cron `/clear` lands in the cron brain's pane (incident
        # 2026-09-19,). The trailing `:` is what
        # makes the same string valid for pane/window commands too: verified
        # live on tmux 3.2a that `=name` alone is rejected by capture-pane/
        # send-keys/paste-buffer/pipe-pane/set-option ("can't find pane") and
        # renders empty in display-message, while `=name:` works for every
        # command used here (has-session, kill-session, rename-session,
        # set-option, resize-window, display-message, capture-pane,
        # send-keys, paste-buffer, pipe-pane), resolves the ACTIVE window
        # under base-index 1, and fails cleanly when only a prefix-sibling
        # exists.
        self._target = exact_target(session_name)
        self._pane_log = self.runtime_dir / "pane.log"
        self._ready_flag = self.runtime_dir / "ready"
        self._inflight = self.runtime_dir / "inflight"
        # /reset's "stop whatever you are doing" signal — a FILE, so an ask()
        # in another process (the nightly pipeline) hears it too.
        self._abort_stamp = self.runtime_dir / ABORT_STAMP_NAME
        self._pane_lock = self.runtime_dir / "pane.lock"
        # Separate lock for handled_rids/pending_orphans bookkeeping — see
        # _state_locked(). Deliberately NOT pane.lock: that one is held for
        # the full duration of a turn (up to DEFAULT_TIMEOUT, an hour), and
        # state writes must never queue up behind it.
        self._state_lock = self.runtime_dir / "state.lock"
        # Owner-facing notices raised by this session (today: a parked pane,
        # see _park_foreign_session), drained by pop_notices() from whichever
        # process talks to the owner first (watchdog tick, chat reply, cron
        # tick). A file, not memory: parking can happen in any of them.
        self._pending_notices = self.runtime_dir / "pending_notices"
        # Labels seen on the pane right after a verified return to its own
        # conversation (see _return_to_main).
        self._returned_labels: set[str] = set()
        # Last marker-pair rid consumed by a completed ask() (wrap=True) OR
        # already forwarded by pop_orphan_replies(). Kept for diagnostics and
        # for older installs; the authoritative dedup store is the SET below.
        self._last_handled_rid = self.runtime_dir / "last_handled_rid"
        # Every rid ever delivered, newest last, capped. A single "last rid"
        # could not survive the pane re-rendering an older pair (the TUI
        # repaints its transcript), which re-promoted an already-delivered
        # reply to "latest" and sent it to the user a second time.
        self._handled_rids = self.runtime_dir / "handled_rids"
        # Rids seen-but-undelivered at the moment a LATER ask() turn marked
        # its own (newer) rid handled directly, bypassing pop_orphan_replies.
        # find_pending_replies()'s watermark rule treats anything above the
        # newest handled pair as "already delivered, however it got there" —
        # true for a repainted pair, false for a genuine orphan that a
        # concurrent turn simply finished ahead of. Queuing them here at the
        # moment they'd otherwise fall below the watermark is what lets
        # pop_orphan_replies() still find them (found in review 2026-08-20).
        self._pending_orphans = self.runtime_dir / "pending_orphans"
        # R1 step 1 (Fable audit): the pinned Claude Code --session-id for
        # the CURRENTLY RUNNING `claude` process in this tmux session — makes
        # the JSONL transcript path deterministic (see _new_session_id,
        # _ensure_locked). Also closes a marker_compliance.py mtime
        # footgun found in production: a caller can read this
        # file instead of guessing by mtime across concurrent sessions.
        self._session_id_file = self.runtime_dir / "session_id"
        # R2a (Fable audit): orphan-salvage stability tracking for an
        # UNCLOSED <<<R:rid>>> span — rid -> last observed extract_open_reply
        # region. In-memory (not persisted): pop_orphan_replies() is only
        # ever called by the single long-lived watchdog process, so a
        # watchdog restart merely costs one extra tick before a candidate is
        # judged stable, never a correctness issue — the alternative (a
        # persisted-but-then-stale entry surviving days) seemed worse.
        self._orphan_open_seen: dict[str, str] = {}

    # ── tmux helpers ─────────────────────────────────────────────────

    def _tmux(
        self, *args: str, input_text: str | None = None
    ) -> subprocess.CompletedProcess:
        proc = self._runner(
            ["tmux", *args],
            capture_output=True,
            text=True,
            check=False,
            input=input_text,
        )
        if proc.returncode != 0:
            logger.warning(
                "tmux %s failed (rc=%s): %s",
                args[0],
                proc.returncode,
                (proc.stderr or "").strip(),
            )
        return proc

    def _capture(self) -> str:
        return self._tmux(
            "capture-pane", "-t", self._target, "-p", "-S", _CAPTURE_SCROLLBACK
        ).stdout

    def _pane_log_size(self) -> int:
        """Byte size of the piped transcript — a growing log is the
        version-proof liveness signal (no on-screen text dependency)."""
        try:
            return self._pane_log.stat().st_size
        except OSError:
            return 0

    def _session_exists(self) -> bool:
        return self._tmux("has-session", "-t", self._target).returncode == 0

    def _enforce_geometry(self) -> bool:
        """Re-assert the pane size on an ALREADY RUNNING session.

        `new-session -x/-y` only sizes the pane at birth. tmux then shrinks the
        window to the smallest ATTACHED client and keeps that size after it
        detaches, so one `dbrain attach` from an 80-column terminal permanently
        narrowed the brain — measured live on 2026-08-20: the brain pane was
        80x23 while the (never attached) cron pane was still 200x50.

        Two user-visible bugs came from that single fact: the TUI hard-wrapped
        replies at ~78 columns, and those wraps reached Telegram as real line
        breaks mid-sentence; and a 23-row pane made the bottom-18-line "chrome"
        window cover the model's own text, so ordinary prose tripped the
        rate-limit signature. `window-size manual` stops clients from resizing
        it again. Cheap and idempotent: only resizes when the size differs.
        Returns True when it did resize (the TUI then re-renders).
        """
        geometry = "#{window_width}x#{window_height}"
        got = self._tmux(
            "display-message", "-p", "-t", self._target, geometry
        ).stdout.strip()
        want = f"{self._pane_width}x{self._pane_height}"
        if got == want:
            return False
        logger.info("resizing pane %s: %s → %s", self.session_name, got or "?", want)
        self._tmux("set-option", "-t", self._target, "window-size", "manual")
        self._tmux(
            "resize-window",
            "-t",
            self._target,
            "-x",
            self._pane_width,
            "-y",
            self._pane_height,
        )
        return True

    def _send_enter(self) -> None:
        self._tmux("send-keys", "-t", self._target, "Enter")

    # ── file locks ───────────────────────────────────────────────────

    @contextmanager
    def _locked(self, *, blocking: bool = True):
        """The single pane lock (see module docstring). Held for the
        duration of a whole turn — never use this for bookkeeping writes,
        see _state_locked() below."""
        fd = os.open(self._pane_lock, os.O_CREAT | os.O_RDWR, 0o644)
        acquired = False
        try:
            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            fcntl.flock(fd, flags)
            acquired = True
            yield True
        except BlockingIOError:
            yield False
        finally:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @contextmanager
    def _state_locked(self):
        """Cross-process critical section for handled_rids / pending_orphans.

        _atomic_write (tmp+rename) is atomic for the WRITE itself, but the
        read→modify→write sequence around it is not: the bot's ask() and the
        watchdog's pop_orphan_replies() both mutate this state from separate
        processes, and an interleaved read-modify-write between the two is a
        classic lost update — one writer's rid never lands, and the reply
        gets redelivered next tick (found in review 2026-08-20).
        Deliberately a SEPARATE lock file from pane.lock: that lock is held
        for the full duration of a turn (up to DEFAULT_TIMEOUT, an hour), and
        state bookkeeping must never queue up behind it. rename() itself
        stays the guard against a write torn by SIGKILL (delivery_guard's
        restart); this lock only guards the read-modify-write sequence.
        """
        fd = os.open(self._state_lock, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # ── lifecycle ────────────────────────────────────────────────────

    def _start_command(self, session_id: str) -> str:
        # R1 step 1 (Fable audit): --session-id pins the JSONL transcript
        # path to a UUID we control instead of one we'd have to discover
        # after the fact. Verified against the installed CLI (2.1.232)
        # before relying on it, per the audit's own caveat that this needed
        # checking rather than trusting: `claude --help` lists
        # `--session-id <uuid>` ("Use a specific session ID for the
        # conversation, must be a valid UUID").
        parts = [
            shlex.quote(self.claude_bin),
            "--dangerously-skip-permissions",
            "--session-id",
            session_id,
        ]
        if self.mcp_config:
            parts += ["--mcp-config", shlex.quote(str(self.mcp_config))]
        if self.system_prompt_file:
            parts += [
                "--append-system-prompt",
                f'"$(cat {shlex.quote(str(self.system_prompt_file))})"',
            ]
        if self.model:
            parts += ["--model", shlex.quote(self.model)]
        # A tmux server started earlier without CLAUDE_CONFIG_DIR (an old
        # doctor run, a plain `tmux`) does not pass the client's environment
        # to new sessions: pin it on the command line so the brain always
        # uses the same login and first-run state as the services.
        config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
        env = f"CLAUDE_CONFIG_DIR={shlex.quote(config_dir)} " if config_dir else ""
        return f"cd {shlex.quote(str(self.work_dir))} && " + env + " ".join(parts)

    # ── R1: pinned session id / transcript path ─────────────────────────

    def _read_session_id(self) -> str | None:
        try:
            v = self._session_id_file.read_text().strip()
            return v or None
        except OSError:
            return None

    def _new_session_id(self) -> str:
        """Generate and persist a FRESH session id — called only when about
        to start a brand-new `claude` process (see _ensure_locked). A
        pre-existing tmux session (the common case: the bot process merely
        restarted) keeps whatever id was pinned when its `claude` process
        actually started, since that is the process still writing the
        transcript this id must keep pointing at."""
        new_id = str(uuid.uuid4())
        if not self._atomic_write(self._session_id_file, new_id + "\n"):
            logger.error(
                "could not persist new session id %s — transcript-dependent "
                "features (the reply source itself, the R3 context check, "
                "marker_compliance) will be unable to resolve the live "
                "transcript until this succeeds",
                new_id,
            )
        return new_id

    def current_transcript_path(self) -> Path | None:
        """Path to the pinned session's JSONL transcript, or None if no
        session has ever been started here. Used by the R3 context-size
        watchdog check, the doctor's periodic marker-compliance check (R7),
        and scripts/marker_compliance.py's default resolution."""
        sid = self._read_session_id()
        if sid is None:
            return None
        return transcript_path(self.work_dir, sid)

    def last_reply_for_resend(self) -> tuple[str, str | None]:
        """Status + optional body for the ``/resend`` command: a manual,
        READ-ONLY escape hatch that re-sends the last
        assistant reply straight from the JSONL transcript when the normal
        tmux-pane-scrape delivery path silently drops it (~3.3% of turns
        measured with no closing marker ever appearing). This is a NEW,
        separate, read-only path — it must never WAIT on ``pane.lock``: the
        whole point is that ``/resend`` keeps working while the main
        session is mid-turn and ``ask()`` holds that lock for up to an hour
        (DEFAULT_TIMEOUT). It still calls ``is_turn_active()`` below, which
        briefly attempts a non-blocking (``LOCK_NB``) acquire when the pane
        is idle — that is a genuine, near-instant acquire+release, not a
        contradiction of the contract: it never *blocks*, and it never
        contends with ``ask()``'s blocking acquire (the two can only ever
        disagree about who got there first, never queue behind each other).

        Returns ``(status, body)`` with status one of:
          - ``"unavailable"`` — no session has ever been started here, or
            the transcript could not be read.
          - ``"empty"`` — no reply history found yet.
          - ``"in_progress"`` — the most recent turn is genuinely still
            being worked on right now (pane.lock held; never reported as
            "lost").
          - ``"no_markers"`` — F-1 fix, round 2 (2026-08-22): the most
            recent assistant text record carries no ``<<<R:id>>>`` marker
            at all, AND no turn is active — so this is not "still
            working", it's a finished reply that never got a marker (or
            one split across transcript records, e.g. the open marker
            landed in an earlier record than the one inspected). This is
            literally the feature's primary target bucket (the audit's
            "3.3% of turns with no marker at all"), so it must never be
            reported as "in_progress" when the session is actually idle.
            There is no marker span to recover a body from here — ``body``
            is always ``None`` for this status; the fix is honesty of the
            status, not recovery.
          - ``"ready"`` — ``body`` is the last complete reply.
        """
        path = self.current_transcript_path()
        if path is None:
            return "unavailable", None
        status, body = latest_reply(path)
        if status == "in_progress":
            # F-1 fix (round 2 review): latest_reply() reports "in_progress"
            # both for a turn that's genuinely still generating AND for the
            # case where the most recent assistant record simply has no
            # marker at all while the session is actually idle (including a
            # reply whose open marker landed in an EARLIER transcript record
            # than the one latest_reply() inspected — it only ever looks at
            # the single most recent assistant text record, so that shape
            # looks identical to "no marker yet" from here). latest_reply()
            # stays pure/lock-free and cannot tell these apart; liveness is
            # decided HERE, exactly mirroring the "unclosed" branch below.
            try:
                active = self.is_turn_active()
            except Exception:
                logger.warning(
                    "last_reply_for_resend: is_turn_active() check failed",
                    exc_info=True,
                )
                # Unknown liveness: fail toward the conservative "still
                # working" verdict rather than falsely claim idleness.
                active = True
            if active:
                return "in_progress", None
            return "no_markers", None
        if status == "unclosed":
            # F1 fix: this is the literal shape of every real delivery-loss
            # incident on record — a closed <<<R:id>>>/<<<E:id>>> pair never
            # appeared, but the finished reply text is sitting right there.
            # latest_reply() stays pure/lock-free and cannot itself tell
            # "still typing" from "done, marker just never landed" —
            # liveness is the only thing that can, and it is decided HERE,
            # not in transcript.py. Mirrors the salvage notice
            # pop_orphan_replies() already prepends for the analogous
            # pane-scrape case (see chat_session.py).
            try:
                active = self.is_turn_active()
            except Exception:
                logger.warning(
                    "last_reply_for_resend: is_turn_active() check failed",
                    exc_info=True,
                )
                # Unknown liveness: fail toward the conservative "still
                # working" verdict rather than risk delivering a reply that
                # might still be mid-write.
                active = True
            if active:
                return "in_progress", None
            notice = "⚠️ <i>ответ восстановлен без закрывающего маркера</i>"
            return "ready", f"{notice}\n\n{body}"
        if status != "empty":
            return status, body
        # The transcript shows no reply history at all — but if a turn is
        # genuinely active right now, "still working" is the honest answer,
        # not "empty" (something IS happening; the transcript just hasn't
        # produced any assistant text yet, e.g. the model is still reading
        # files). is_turn_active() only ever takes pane.lock NON-blocking
        # (LOCK_NB — it never waits for it), but the call is still wrapped
        # defensively here: a failure in it must never take this read-only,
        # user-triggered path down with it — fall back to the transcript's
        # own "empty" verdict instead of letting anything propagate.
        try:
            if self.is_turn_active():
                return "in_progress", None
        except Exception:
            logger.warning(
                "last_reply_for_resend: is_turn_active() check failed",
                exc_info=True,
            )
        return "empty", None

    def _transcript_project_dir(self) -> Path:
        return transcript_dir(self.work_dir)

    def _resync_session_id_after_clear(self, before: set[Path]) -> None:
        """Re-pin the session id after `/clear` (see clear()'s docstring for
        why this is necessary at all).

        Waits for a jsonl file to appear that did NOT exist in ``before`` —
        NOT simply "the newest jsonl", because the cron brain
        (get_cron_session) shares this exact SAME work_dir/project directory
        (verified in runtime.py), so a concurrently-active cron turn could
        otherwise be mistaken for the just-cleared session's new transcript.
        """
        project_dir = self._transcript_project_dir()
        deadline = self._clock() + 10.0
        while self._clock() < deadline:
            try:
                after = set(project_dir.glob("*.jsonl"))
            except OSError:
                after = set()
            new_files = after - before
            if new_files:
                newest = max(new_files, key=lambda p: p.stat().st_mtime)
                if self._atomic_write(self._session_id_file, newest.stem + "\n"):
                    logger.info("session id resynced after /clear -> %s", newest.stem)
                else:
                    logger.error(
                        "found the post-/clear transcript (%s) but could not "
                        "persist the resynced session id",
                        newest.stem,
                    )
                return
            self._sleep(self._poll_interval)
        logger.warning(
            "could not resync session id after /clear within 10s — "
            "transcript-dependent features will keep pointing at the "
            "pre-clear transcript, which has stopped growing"
        )

    def _ensure_locked(self) -> str | None:
        """Create + ready the session if needed. Caller must hold the lock.

        An existing session is reused only while its pane shows the bot's own
        conversation. A pane switched to a background task via the
        "← N agents" view (agent-infra: it sat on
        `night-second-brain` after two weeks of human use while the bot ran
        on Codex) still looks READY but answers into a transcript the bot
        never reads, so every ask() times out. Conservative order — the
        bot's conversation is only given up when nothing else worked:

        1. no foreign view on screen → reuse as is;
        2. otherwise press the keys that return to the main conversation
           (verified live) and reuse the pane if that visibly worked;
        3. only then park the session (renamed, owner notified) and start a
           fresh one by the normal path below.

        Returns the capture taken for that check when the existing session
        was reused (ask() uses it as its pre-send frame), else None."""
        if self._session_exists():
            # Geometry first, frame second: a resize makes the TUI re-render,
            # so a frame taken before it (80 columns, 23 rows) is neither
            # what the view check should judge nor what ask() should use as
            # its pre-send frame (review 2026-09-19).
            if self._enforce_geometry():
                self._sleep(self._poll_interval)
            cap = self._capture()
            view = self._foreign_view(cap)
            if view is not None:
                back = self._return_to_main(view)
                if back is not None:
                    cap, view = back, None
            if view is None:
                self._ensure_pipe()
                return cap
            if not self._park_foreign_session(view):
                # Neither rename nor kill worked and the stuck pane is still
                # there: creating a session under the same name would fail
                # anyway, and reusing it is the very bug this guards against.
                # Keep the pinned session_id (the stuck process still owns
                # it), drop the ready flag and fail loudly instead.
                self._ready_flag.unlink(missing_ok=True)
                raise RuntimeError(
                    f"session {self.session_name} is stuck on a foreign view "
                    f"({view!r}) and could be neither parked nor killed"
                )
        self._ready_flag.unlink(missing_ok=True)
        # R4 fix (Fable audit, verified live 2026-08-22 on the installed
        # tmux 3.2a): history-limit is fixed at WINDOW CREATION time. The
        # previous code set it via `set-option -t <session>` AFTER
        # `new-session`, which is a silent no-op for the window just
        # created — a throwaway session reproduced this exactly (the
        # post-creation call left history_limit at tmux's own default,
        # 2000, never 50000). Only `-g` (global default), applied BEFORE
        # `new-session`, is honored by a window created afterward — verified
        # the same way. Bigger scrollback so long replies stay within
        # capture range; global is safe here since this process is the only
        # thing creating dbrain tmux sessions.
        self._tmux("set-option", "-g", "history-limit", "50000")
        # R1 step 1: a brand-new `claude` process needs a brand-new pinned
        # session id — see _new_session_id's docstring for why this is only
        # done on THIS branch (session doesn't exist yet), never when an
        # existing tmux session is merely being re-attached to.
        session_id = self._new_session_id()
        new_session_args = [
            "new-session",
            "-d",
            "-s",
            self.session_name,
            "-x",
            self._pane_width,
            "-y",
            self._pane_height,
            self._start_command(session_id),
        ]
        # B4 fix: on a COLD start (no tmux server yet) the `-g set-option`
        # above silently fails before any server exists to hold the global
        # default, so `new-session` below would fall back to tmux's own
        # built-in history-limit (2000) — see tmux_config's docstring in
        # __init__ for the live-verified repro. `-f` on THIS SAME invocation
        # makes tmux read deploy/tmux.conf (which sets the same `-g
        # history-limit 50000`) as part of starting the new server, before
        # the session is created — verified live to produce history_limit
        # =50000 from cold. Harmless on a warm start: an already-running
        # server ignores a client's `-f`, since config files are only read
        # at server startup — so the `-g set-option` above remains the
        # thing actually doing the work in that (more common) case.
        if self.tmux_config is not None and self.tmux_config.exists():
            self._tmux("-f", str(self.tmux_config), *new_session_args)
        else:
            if self.tmux_config is not None:
                logger.warning(
                    "tmux config %s not found — history-limit relies solely "
                    "on the -g set-option above, which is a no-op on a cold "
                    "tmux server",
                    self.tmux_config,
                )
            self._tmux(*new_session_args)
        self._attach_pipe()

        deadline = self._clock() + self._startup_timeout
        last_state: PaneState | None = None
        while self._clock() < deadline:
            cap = self._capture()
            state = classify_state(cap)
            # Debounce: only Enter on the transition INTO the trust prompt,
            # never on every poll (avoids stray blank submissions).
            if state == PaneState.TRUST_PROMPT:
                if last_state != PaneState.TRUST_PROMPT:
                    self._send_enter()
                last_state = state
                self._sleep(self._poll_interval)
                continue
            # Bypass-permissions accept screen (fresh config dir): unlike TRUST,
            # the safe default ❯ sits on "1. No, exit", so we must actively pick
            # "2. Yes, I accept". Debounced to the transition like TRUST.
            if state == PaneState.BYPASS_PROMPT:
                if last_state != PaneState.BYPASS_PROMPT:
                    self._tmux("send-keys", "-t", self._target, "2")
                    self._send_enter()
                last_state = state
                self._sleep(self._poll_interval)
                continue
            if state == PaneState.READY:
                self._ready_flag.write_text("ready\n")
                logger.info("Claude session %s is ready", self.session_name)
                return None
            last_state = state
            self._sleep(self._poll_interval)
        raise RuntimeError(
            f"session {self.session_name} not ready in {self._startup_timeout}s; "
            f"last state={last_state}; pane tail:\n"
            + "\n".join(self._capture().splitlines()[-8:])
        )

    def _attach_pipe(self) -> None:
        # Pre-create the transcript owner-only: `cat >>` appends and keeps
        # the mode, while letting tmux create it would use the server umask.
        self._pane_log.touch()
        os.chmod(self._pane_log, 0o600)
        self._tmux(
            "pipe-pane",
            "-t",
            self._target,
            f"cat >> {shlex.quote(str(self._pane_log))}",
        )

    def _ensure_pipe(self) -> None:
        """Re-attach pane.log to an existing session that lost it (e.g. one
        started by hand). pane.log growth is the version-proof liveness
        signal; without it every quiet turn reads as a stall. Only an
        explicit "0" re-pipes — an unreadable answer leaves things alone."""
        piped = self._tmux(
            "display-message", "-p", "-t", self._target, "#{pane_pipe}"
        ).stdout.strip()
        if piped == "0":
            logger.warning(
                "session %s has no pane.log pipe — re-attaching", self.session_name
            )
            self._attach_pipe()

    # ── foreign view handling ────────────

    def _own_titles(self) -> set[str]:
        """Every name the pinned conversation has carried (``/rename``, the
        CLI's auto-title) — read from its own JSONL transcript. Empty when
        there is no pinned transcript or it cannot be read."""
        path = self.current_transcript_path()
        if path is None:
            return set()
        titles: set[str] = set()
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if '-title"' not in line and '"agent-name"' not in line:
                        continue  # cheap pre-filter: transcripts can be large
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    field = _TITLE_FIELDS.get(rec.get("type"))
                    value = rec.get(field) if field else None
                    if isinstance(value, str) and value.strip():
                        titles.add(value.strip())
        except OSError:
            return set()
        return titles

    def _foreign_view(self, cap: str) -> str | None:
        """What the pane shows instead of the bot's conversation, or None.

        A labelled input box alone is not enough: ``/rename foo`` in the
        bot's own conversation draws exactly the same "──… foo ─" border as
        a background task's view (verified live on Claude Code 2.1.278). A
        label that is one of the pinned conversation's own titles is
        therefore the bot's conversation. Any other label is only a
        SUSPICION — _return_to_main() decides the rest."""
        if is_agents_list_view(cap):
            return AGENTS_LIST_VIEW
        label = foreign_view_label(cap)
        if label is None:
            return None
        titles = self._own_titles() | self._returned_labels
        stem = label[:-1].rstrip() if label.endswith("…") else None
        if label in titles or (stem and any(t.startswith(stem) for t in titles)):
            return None
        return label

    def _wait_for_view(self, want_list: bool) -> str | None:
        """Poll until the agents list is (or is no longer) on screen."""
        deadline = self._clock() + self._view_switch_timeout
        while True:
            cap = self._capture()
            if is_agents_list_view(cap) == want_list:
                return cap
            if self._clock() >= deadline:
                return None
            self._sleep(self._poll_interval)

    def _clear_input_draft(self) -> bool:
        """Empty the input box of whatever view is on screen, keystroke-safe.

        With text in the box, ``Left`` only moves the cursor (verified live,
        2.1.278) and Escape in the agents list only clears the draft ("esc
        to clear") — and Escape in a background task's view INTERRUPTS that
        task's running turn, so Escape is never the way to clear. ``C-u``
        alone deletes only up to the cursor; ``C-k`` + ``C-u`` clears the
        current line wherever the cursor is, and repeating it eats a
        multi-line draft line by line (3 rounds for 3 lines, cursor mid-
        text, live). On an empty box, in a running turn and in the list,
        both keys do nothing (live). Plain capture cannot tell a dim
        placeholder from typed text, so the stop signal is "a round changed
        nothing" — compared row by row, empty rows included: after the last
        row is emptied the box still has that row, and only the next round
        removes it (live: a stripped comparison stopped there and left the
        first row of a two-row draft behind). Whatever is left anyway makes
        the next step's on-screen check fail, and the session is parked.
        False only when it never settled."""
        box = input_box_text(self._capture())
        if box is None:
            return True  # no box to clear; the next step's check decides
        for _ in range(_MAX_CLEAR_ROUNDS):
            self._tmux("send-keys", "-t", self._target, "C-k")
            self._tmux("send-keys", "-t", self._target, "C-u")
            self._sleep(self._paste_settle)
            now = input_box_text(self._capture())
            if now == box:
                return True
            logger.info(
                "session %s: cleared a draft from the input box", self.session_name
            )
            box = now
        return False

    def _return_to_main(self, view: str) -> str | None:
        """Try to bring the pane back to the bot's own conversation.

        Verified live on Claude Code 2.1.278 in an isolated tmux server:
        from a background task's view, ``Left`` on the empty input opens the
        "← N agents" list, and ``Escape`` there returns to the session's OWN
        conversation (the list itself says "esc returns to it"); from the
        list alone, ``Escape`` suffices. Both steps are confirmed on screen
        before the next one — sent back to back, the TUI swallows the Escape.

        Returns a fresh capture of the conversation when both steps visibly
        happened, else None (then the caller parks the session). A pane that
        was in fact on the renamed main conversation all along (label not in
        the pinned transcript, e.g. after the CLI re-keyed it) makes the same
        round trip and lands where it started — harmless.

        Any draft in the input box is cleared first (_clear_input_draft):
        with text there neither key does what is described above."""
        if not self._clear_input_draft():
            logger.warning(
                "session %s: could not clear the input box of view %r",
                self.session_name,
                view,
            )
            return None
        if view != AGENTS_LIST_VIEW:
            # Up to two Lefts. Verified live (2.1.278): once the box has been
            # edited at all — a draft cleared by _clear_input_draft, or text
            # a human typed and deleted again — the first Left on the now
            # empty box does nothing visible and only the second opens the
            # list; on a box untouched since the view opened one suffices.
            # The second is sent only after the first visibly failed, and
            # Left on an empty box outside the list moves nothing.
            for _ in range(2):
                self._tmux("send-keys", "-t", self._target, "Left")
                if self._wait_for_view(want_list=True) is not None:
                    break
            else:
                logger.warning(
                    "session %s: view %r did not open the agents list on Left",
                    self.session_name,
                    view,
                )
                return None
        self._tmux("send-keys", "-t", self._target, "Escape")
        cap = self._wait_for_view(want_list=False)
        if cap is None:
            logger.warning(
                "session %s: Escape did not leave the agents list", self.session_name
            )
            return None
        label = foreign_view_label(cap)
        if label is not None:
            # Where "esc returns to it" lands IS the session's own
            # conversation — remember its name so the next turn does not
            # repeat the round trip (its transcript may be under an id the
            # pinned session_id no longer points at).
            self._returned_labels.add(label)
        logger.warning(
            "session %s was on a foreign view (%r) — returned to its own "
            "conversation (now labelled %r)",
            self.session_name,
            view,
            label,
        )
        return cap

    def _park_foreign_session(self, label: str) -> bool:
        """Move a session whose pane shows a foreign view out of the bot's way.

        Renamed, not killed, so whatever a human was doing in it survives.
        The parked name does not start with ``dbrain_``, so the nightly
        server cleanup (which never touches ``dbrain_*``) can reclaim it once
        it has sat idle and detached long enough. Its pane.log pipe is closed
        so it can never write into the bot's transcript again — and if that
        close fails the parked session is killed instead, since its output
        growing pane.log would mask a stall of the fresh session. Kill is
        also the fallback when the rename itself fails. The owner is told
        either way: the bot's conversation context is gone.

        Returns False only when neither rename nor kill succeeded.
        """
        parked = f"parked_{self.session_name}_{time.strftime('%Y%m%d-%H%M%S')}"
        logger.warning(
            "session %s shows a foreign view (%r), not the bot's conversation "
            "— parking it as %s and starting a fresh one",
            self.session_name,
            label,
            parked,
        )
        if self._tmux("rename-session", "-t", self._target, parked).returncode == 0:
            if self._tmux("pipe-pane", "-t", exact_target(parked)).returncode != 0:
                logger.error(
                    "could not close pane.log pipe of parked %s — killing it", parked
                )
                if self._tmux("kill-session", "-t", exact_target(parked)).returncode:
                    logger.error("could not kill parked %s either", parked)
                    how = (
                        f"отложена как {parked}, но отключить её вывод и "
                        "закрыть её не удалось — её стоит закрыть вручную"
                    )
                else:
                    how = f"закрыта (отложить не удалось), имя было {parked}"
            else:
                how = f"отложена как {parked}"
            self._prune_parked()
        elif (
            self._tmux("kill-session", "-t", self._target).returncode == 0
            or not self._session_exists()  # vanished on its own meanwhile
        ):
            how = "закрыта (переименовать не удалось)"
        else:
            logger.error(
                "session %s: neither rename nor kill worked", self.session_name
            )
            self._queue_notice(
                f"🔴 Сессия {self.session_name} застряла на экране фоновой задачи "
                f"«{label}» и её не удалось ни отложить, ни закрыть — ответы "
                "не будут доходить. Нужен dbrain repair."
            )
            return False
        kind = "кроновая" if self.session_name.endswith("_cron") else "основная"
        self._queue_notice(
            f"⚠️ Контекст разговора сброшен: {kind} сессия {self.session_name} "
            f"была на экране фоновой задачи «{label}», {how}. "
            "Дальше отвечает свежая сессия."
        )
        return True

    def _prune_parked(self) -> None:
        """Keep at most _MAX_PARKED parked sessions of THIS brain; kill the
        oldest. The pattern is anchored on the timestamp so the main brain
        never touches its ``_cron`` sibling's parked sessions."""
        listing = self._tmux("list-sessions", "-F", "#{session_name}")
        if listing.returncode != 0:
            return
        name = re.escape(self.session_name)
        pattern = re.compile(rf"^parked_{name}_\d{{8}}-\d{{6}}$")
        parked = sorted(n for n in (listing.stdout or "").split() if pattern.match(n))
        for name in parked[:-_MAX_PARKED]:
            logger.info("killing old parked session %s", name)
            self._tmux("kill-session", "-t", exact_target(name))

    # ── owner notices ────────────────────────────────────────────────

    def _queue_notice(self, text: str) -> None:
        try:
            with self._state_locked(), self._pending_notices.open("a") as fh:
                fh.write(json.dumps(text, ensure_ascii=False) + "\n")
        except OSError:
            logger.error("could not queue owner notice: %s", text)

    def requeue_notices(self, texts: list[str]) -> None:
        """Put notices whose delivery failed back at the FRONT of the queue
        (they are older than anything queued meanwhile). Raises OSError when
        the queue cannot be written — the caller then logs them as lost."""
        if not texts:
            return
        with self._state_locked():
            try:
                newer = self._pending_notices.read_text(encoding="utf-8")
            except FileNotFoundError:
                newer = ""
            back = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in texts)
            if not self._atomic_write(self._pending_notices, back + newer):
                raise OSError(f"could not rewrite {self._pending_notices}")

    def pop_notices(self) -> list[str]:
        """Owner-facing notices raised since the last call, oldest first.
        Each is handed out once, to whichever caller asks first."""
        with self._state_locked():
            try:
                raw = self._pending_notices.read_text(encoding="utf-8")
            except OSError:
                return []
            self._pending_notices.unlink(missing_ok=True)
        out: list[str] = []
        for line in raw.splitlines():
            try:
                text = json.loads(line)
            except ValueError:
                continue
            if isinstance(text, str) and text:
                out.append(text)
        return out

    def ensure_session(self) -> None:
        with self._locked() as got:
            if got:
                self._ensure_locked()

    def is_healthy(self) -> bool:
        return self._session_exists()

    def current_state(self) -> PaneState:
        """Classify the live pane (for the watchdog / doctor)."""
        return classify_state(self._capture())

    def is_working(self) -> bool:
        """True iff the pane shows an active turn (for the watchdog).

        Change-aware across successive calls to THIS method: the PROGRESS
        signatures (elapsed-timer, background-agent wait) only count when
        the chrome differs from the previous call's capture. Without this a
        session frozen mid-turn (dead subagent, a process stuck in D-state)
        holds one of those signatures on its last rendered frame forever,
        which reports "working" on every watchdog tick and disarms the hang
        detector entirely — found live in review 2026-08-20. The legacy
        "esc to interrupt" hint stays exempt (see is_working_progressing) —
        but only for STATIC_TRUST_WINDOW seconds of a genuinely frozen pane
: the hint lives in a footer this CLI shows whenever
        anything at all is interruptible, so trusting it forever left the hang
        detector disarmed in exactly the UNKNOWN-state corner it exists for.
        The window is reset by EITHER real signal — a chrome change or
        pane.log growth. The pane.log half is not optional: strip_reply_bodies()
        removes a streaming reply from chrome, so a legitimately live turn can
        hold chrome byte-identical across polls while the log grows; without
        that reset this would be a fresh B3-class regression (interrupting
        live, quiet work).
        """
        cap = self._capture()
        now = self._clock()
        log_size = self._pane_log_size()
        prev = self._last_is_working_cap
        moved = (
            prev is None
            or _chrome(prev) != _chrome(cap)
            or log_size > self._last_is_working_log_size
        )
        if moved:
            self._static_frozen_since = None
        elif self._static_frozen_since is None:
            self._static_frozen_since = now
        trust_static = (
            self._static_frozen_since is None
            or now - self._static_frozen_since < STATIC_TRUST_WINDOW
        )
        result = is_working_progressing(cap, prev, trust_static=trust_static)
        self._last_is_working_cap = cap
        self._last_is_working_log_size = log_size
        return result

    def _confirmed_rate_limited(self, first_cap: str, log_id: str) -> bool:
        """True iff RATE_LIMITED persists across several consecutive polls.

        See _RATE_LIMIT_CONFIRM_POLLS. ``first_cap`` is the capture already
        classified by the caller (avoids a redundant capture-pane call);
        subsequent samples are fresh captures spaced by poll_interval.
        """
        if classify_state(first_cap) != PaneState.RATE_LIMITED:
            return False
        for attempt in range(_RATE_LIMIT_CONFIRM_POLLS - 1):
            self._sleep(self._poll_interval)
            if classify_state(self._capture()) != PaneState.RATE_LIMITED:
                # The signature was there but did not persist — exactly the
                # false positive this confirm-poll loop exists to catch.
                # Logged (not silent) since after deploy this is the one
                # signal that shows the narrowed _RATE_RE is doing its job,
                # rather than just never seeing a suspicious frame at all
                # (round-2 finding, 2026-08-20).
                logger.info(
                    "rate-limit signature for %s did not persist past poll "
                    "%d/%d — suppressed as a false positive",
                    log_id,
                    attempt + 1,
                    _RATE_LIMIT_CONFIRM_POLLS,
                )
                return False
        logger.warning(
            "rate-limited (confirm-polls) for %s: signature persisted across %d polls",
            log_id,
            _RATE_LIMIT_CONFIRM_POLLS,
        )
        return True

    def capture_text(self) -> str:
        """Raw pane text (read-only, same as every other introspection
        method — capture-pane is a read, never guarded by the pane lock)."""
        return self._capture()

    # ── delivered-rid bookkeeping ────────────────────────────────────

    _HANDLED_CAP = 200  # plenty of history; the pane holds far fewer pairs

    def _read_handled(self) -> list[str]:
        try:
            return self._handled_rids.read_text().split()
        except OSError:
            return []

    def _atomic_write(self, target: Path, payload: str) -> bool:
        """tmp-then-rename write. A bare ``write_text`` here is a
        truncate-then-write with no lock: two writers (the bot's ask() and
        the watchdog's pop_orphan_replies both call _mark_handled) can
        interleave, and a process killed mid-write (delivery_guard's restart
        sends SIGKILL after TimeoutStopSec) can leave the file empty or
        truncated — which then reads as watermark=-1 and replays the whole
        capture window as "never delivered" (found in review 2026-08-20)."""
        try:
            tmp = target.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(payload)
            os.replace(tmp, target)
            return True
        except OSError as exc:
            logger.error("could not write %s: %s", target, exc)
            return False

    def _mark_handled(self, rid: str) -> bool:
        """Record a rid as delivered (idempotent, newest last, capped).

        Returns True only once the write is durable. Callers that hand a
        reply to the user BEFORE calling this (pop_orphan_replies) must
        treat a False return as "do not deliver": the store exists precisely
        so a rid is never handed out twice, and delivering on top of a
        failed write reopens that hole while also going silent about it —
        this used to log at WARNING and hand the reply out regardless.
        """
        with self._state_locked():
            rids = [r for r in self._read_handled() if r != rid]
            rids.append(rid)
            payload = "\n".join(rids[-self._HANDLED_CAP :]) + "\n"
            if not self._atomic_write(self._handled_rids, payload):
                return False
            try:
                self._last_handled_rid.write_text(rid)
            except OSError as exc:
                # Diagnostics-only mirror of the rid; the authoritative store
                # above already landed, so this alone must not block delivery.
                logger.warning("could not update last_handled_rid to %s: %s", rid, exc)
            return True

    def _ensure_migrated(self, cap: str) -> bool:
        """Idempotent migration point: adopt every pair already on screen as
        delivered, on an install that predates the ``handled_rids`` store.

        MUST be called at the START of both ``ask()``'s completion path and
        ``pop_orphan_replies()``, before either reads or writes
        ``handled_rids``/``pending_orphans``. Without this, whichever of the
        two runs FIRST after an upgrade sees an empty ``handled_rids``,
        treats every already-visible reply as "unhandled", and (in ask()'s
        case) unconditionally queues the whole scrollback into
        ``pending_orphans`` — which pop_orphan_replies() then hands out on
        its own next tick, bypassing the watermark entirely and flushing the
        pane's full history into the chat (found re-driving this plan,
        2026-08-20; ask() alone calling _mark_handled for its OWN rid used to
        make handled_rids "exist" without ever having been properly seeded,
        which permanently disabled the old migration check inside
        pop_orphan_replies()).

        Detected by the OLD file (``last_handled_rid``) existing while the
        new one does not — a genuinely fresh runtime dir has neither, and
        there anything on the pane really is undelivered, so no migration
        runs. Also re-triggers when ``handled_rids`` DOES exist but does not
        contain the rid ``last_handled_rid`` mirrors: under normal operation
        ``_mark_handled`` writes both files together for every rid it
        records, so the two can only disagree if ``handled_rids`` was left
        behind by something else — an aborted/partial deploy, in the exact
        state found re-driving this plan against the live target
        (2026-08-20): a stray ``handled_rids`` from a botched prior deploy
        sat on disk, 4.5h stale and missing the rid ``last_handled_rid`` had
        since moved on to, which meant every one of the (already-delivered)
        pairs still on screen looked "unhandled" and would have been resent.
        Treated the same as the fresh-install case: reseed from the union of
        whatever ``handled_rids`` already has, everything visible on the
        pane right now, and the legacy rid — this only ever ADDS rids, never
        drops one, so it cannot un-deliver something already recorded.
        Returns True iff migration happened just now: the caller must treat
        this tick as delivering/queuing nothing, since ``handled_rids``
        already reflects everything currently on screen.
        """
        with self._state_locked():
            if not self._last_handled_rid.exists():
                return False
            try:
                legacy = self._last_handled_rid.read_text().strip()
            except OSError:
                legacy = ""
            handled_rids_exists = self._handled_rids.exists()
            handled_now = set(self._read_handled()) if handled_rids_exists else set()
            if handled_rids_exists and (not legacy or legacy in handled_now):
                return False
            known = handled_now | reply_rids(cap)
            if legacy:
                known.add(legacy)
            if not self._atomic_write(
                self._handled_rids, "\n".join(sorted(known)) + "\n"
            ):
                logger.error(
                    "could not seed handled_rids during migration (%d rid(s)) "
                    "— will retry next tick",
                    len(known),
                )
                return False
            # Empty pending_orphans in the SAME critical section: a reader
            # between the two writes could otherwise see a populated queue
            # that predates seeding and hand it out unconditionally, which is
            # exactly the bug this method exists to close.
            self._atomic_write(self._pending_orphans, "")
            logger.info(
                "seeded %d pre-existing rid(s) as already delivered", len(known)
            )
            return True

    # ── pending-orphan queue (see _pending_orphans field docstring) ───

    def _read_pending_orphans(self) -> list[str]:
        try:
            return self._pending_orphans.read_text().split()
        except OSError:
            return []

    def _queue_pending_orphans(self, rids: list[str]) -> None:
        if not rids:
            return
        with self._state_locked():
            existing = self._read_pending_orphans()
            merged = existing + [r for r in rids if r not in existing]
            payload = "\n".join(merged[-self._HANDLED_CAP :]) + "\n"
            if not self._atomic_write(self._pending_orphans, payload):
                logger.error(
                    "could not queue %d pending orphan rid(s) — they may be lost",
                    len(rids),
                )

    def _clear_pending_orphans(self, delivered: set[str]) -> None:
        with self._state_locked():
            remaining = [r for r in self._read_pending_orphans() if r not in delivered]
            payload = "\n".join(remaining) + "\n" if remaining else ""
            if not self._atomic_write(self._pending_orphans, payload):
                logger.error(
                    "could not clear %d delivered pending-orphan rid(s) — may "
                    "be redelivered next tick",
                    len(delivered),
                )

    def pop_orphan_replies(self) -> list[str]:
        """Every self-marked reply nobody's ``ask()`` is waiting on, oldest
        first — consumed, so each is returned exactly once.

        Plural on purpose. The single-latest version lost a reply whenever a
        newer turn produced its own marker pair before the poller ran: the
        orphan stopped being "the latest" and no code path ever looked at it
        again. Delivery now depends only on the rid never having been sent,
        not on poll timing.
        """
        if self.is_turn_active():
            return []
        cap = self._capture()
        if self._ensure_migrated(cap):
            return []
        handled = set(self._read_handled())
        out: list[str] = []
        delivered_pending: set[str] = set()
        # Explicitly queued rids (see _pending_orphans docstring) are handed
        # out unconditionally, ignoring find_pending_replies' watermark: that
        # rule assumes anything above the newest handled pair is already
        # delivered, which is false for exactly these — a genuine orphan a
        # later ask() turn happened to finish ahead of.
        for rid in self._read_pending_orphans():
            if rid in handled:
                delivered_pending.add(rid)  # already handled some other way
                continue
            body = extract_reply(cap, rid)
            if body is None:
                # The queued rid's marker pair is no longer intact in the
                # current capture — scrolled out of the window, or the pane
                # repainted over it. Give up on it HERE rather than retry it
                # silently forever: left in pending_orphans, this was
                # re-attempted every tick with nothing in the log to show
                # for it — a genuinely lost reply with zero trace (round-2
                # finding, 2026-08-20).
                logger.error(
                    "queued orphan %s no longer extractable from the pane "
                    "capture — giving up, reply is lost",
                    rid,
                )
                delivered_pending.add(rid)  # stop retrying; see log above
                continue
            if not self._mark_handled(rid):
                logger.error(
                    "could not mark queued orphan %s handled — not delivering", rid
                )
                continue
            delivered_pending.add(rid)
            handled.add(rid)
            logger.info("delivered queued orphan reply rid=%s len=%d", rid, len(body))
            out.append(body)
        self._clear_pending_orphans(delivered_pending)
        for rid, body in find_pending_replies(cap, handled):
            # Mark consumed BEFORE handing it over: an occasional lost
            # delivery beats ever resending the same message on the next tick.
            if not self._mark_handled(rid):
                logger.error("could not mark orphan %s handled — not delivering", rid)
                continue
            logger.info("delivered orphan reply rid=%s len=%d", rid, len(body))
            out.append(body)

        # R2a (Fable audit): rescue an UNCLOSED span nobody's ask() is
        # waiting on — before this, F1 held: the ceiling's log line PROMISED
        # "rid left unhandled for the orphan poller", but this method could
        # only ever recover COMPLETE R/E pairs, so a reply that never got its
        # closing marker AND missed ask()'s own salvage window (R scrolled
        # out of the capture window, no recognizable boundary, region never
        # stabilized within DEFAULT_SALVAGE_STABLE, or the pane still showed
        # a working signature) was lost forever. Bar for delivery here is
        # "two consecutive watchdog ticks show byte-identical content" —
        # cheaper than ask()'s time-based stability window because the
        # watchdog already polls on its own fixed interval (DEFAULT_TICK),
        # so two ticks already imply real elapsed time.
        if main_turn_finished(cap):
            for rid in sorted(open_reply_rids(cap)):
                if rid in handled:
                    self._orphan_open_seen.pop(rid, None)
                    continue
                region = extract_open_reply(cap, rid)
                if region is None:
                    # R scrolled out of the capture window, or no boundary
                    # line was found — nothing stable to compare against yet.
                    self._orphan_open_seen.pop(rid, None)
                    continue
                previous = self._orphan_open_seen.get(rid)
                if previous != region:
                    self._orphan_open_seen[rid] = region
                    continue
                if not self._mark_handled(rid):
                    logger.error(
                        "could not mark salvaged orphan %s handled — not delivering",
                        rid,
                    )
                    continue
                self._orphan_open_seen.pop(rid, None)
                handled.add(rid)
                logger.warning(
                    "salvaged orphan reply rid=%s (unclosed span, stable "
                    "across 2 poller ticks) len=%d",
                    rid,
                    len(region),
                )
                out.append(
                    f"⚠️ <i>ответ восстановлен без закрывающего маркера</i>\n\n{region}"
                )
        return out

    def pop_orphan_reply(self) -> str | None:
        """Return and consume a proactive reply nobody's ``ask()`` is
        waiting on, or ``None`` if there isn't one.

        A background subagent can finish while the session is otherwise
        idle; the CLI resumes the model outside any ``ask()`` call, so if
        the model writes a reply worth surfacing it self-wraps it in a
        fresh ``<<<R:id>>>``/``<<<E:id>>>`` pair per the session contract
        (see deploy/brain-system.md) — there is simply no live caller to
        hand it to otherwise. This is the delivery side: called by the
        watchdog whenever the pane is idle, it looks for the latest
        complete marker pair and returns its body once, the first time it's
        seen.

        Dedup is the ``handled_rids`` set shared with ``ask()``: every rid
        either method hands out is recorded, so a reply already delivered
        through the normal ask()/chat path is never re-delivered here, and an
        orphan already forwarded once is never forwarded twice.
        ``is_turn_active()`` is re-checked to refuse to touch the pane while a
        real ask() might still be mid-turn.

        Thin compatibility wrapper over :meth:`pop_orphan_replies` — prefer
        that one, which cannot silently drop a superseded orphan.
        """
        replies = self.pop_orphan_replies()
        return replies[0] if replies else None

    def force_recover(self) -> bool:
        """Watchdog entry point: take the lock non-blocking; if free, kill and
        recreate. Returns False if a live ask() currently holds the lock."""
        with self._locked(blocking=False) as got:
            if not got:
                return False
            self._tmux("kill-session", "-t", self._target)
            self._ready_flag.unlink(missing_ok=True)
            self._inflight.unlink(missing_ok=True)
            self._ensure_locked()
            return True

    def nudge(self, text: str = "Continue") -> bool:
        """Type a neutral prompt into an idle-but-parked session.

        Claude Code never re-checks a subscription limit on its own: it parks
        at the banner and waits for the next input, so after the reset time
        passes the session sits there indefinitely (2026-08-20 incident — a
        manual send-keys woke it instantly). This is that send-keys, for the
        watchdog to fire once the reset time has passed.

        Deliberately NOT ``steer()``: this must never type into a live turn,
        so it takes the pane lock non-blocking and gives up if a real ask()
        holds it. Returns True when the nudge was actually sent.

        Marks the turn maint-prefixed in ``inflight`` before typing (mirrors
        ask()'s claim) so it is visible to anything reading that file, not
        only to lock-based checks: the lock itself is released the instant
        this ``with`` block exits (nudge never waits for a reply), so a
        message arriving moments later would otherwise see a free lock AND
        no inflight record, and be classified as a brand-new turn — pasting
        a second prompt into the pane on top of the one this just sent.
        """
        with self._locked(blocking=False) as got:
            if not got:
                return False
            self._inflight.write_text(f"{MAINT_PREFIX}nudge\n{self._clock()}\n")
            self._send_text(text)
            self._send_enter()
            return True

    def kill(self) -> None:
        """Tear down the session (lock-guarded). For CLI/teardown use."""
        with self._locked() as got:
            if got:
                self._tmux("kill-session", "-t", self._target)
                self._ready_flag.unlink(missing_ok=True)
                self._inflight.unlink(missing_ok=True)

    # ── steering (concurrent input into a live turn) ─────────────────

    def steer(self, text: str) -> None:
        """Type text into the pane WITHOUT taking the pane lock.

        Used to inject guidance into an in-flight turn (the lock is held by
        that turn's ask()). The interactive TUI feeds mid-turn input to the
        model (steer/queue semantics) — VERIFY-LIVE per CLI version.
        """
        self._send_text(text)
        self._send_enter()

    def interrupt(self) -> None:
        """Stop the current response (TUI-native Escape, no lock).

        C-c is deliberately NOT used here: on an idle prompt a C-c starts the
        double-C-c exit sequence; Escape only ever cancels the response.
        """
        self._tmux("send-keys", "-t", self._target, "Escape")

    def is_turn_active(self) -> bool:
        """True iff a turn is in flight (the pane lock is held)."""
        with self._locked(blocking=False) as got:
            return not got

    def is_pane_turn_active(self) -> bool:
        """True iff the PANE shows the main turn still running, regardless
        of whether ``ask()`` currently holds the pane lock (see
        ``is_turn_active``).

        During an unattended long-running cascade the lock is free (nothing
        called ``ask()`` for this turn) while the pane itself is genuinely
        busy — the exact shape ``long_run.py``/the busy-active classification
        exists to describe. ``chat.py`` uses this so a user's stop-word can
        reach the interrupt path even when the lock-based ``is_turn_active``
        says nothing is in flight (Step D,)."""
        return is_main_turn_active(self.capture_text())

    def activity_fingerprint(self) -> tuple[str, int]:
        """What changes whenever the pane is alive: its chrome and the size
        of the piped transcript. Two equal fingerprints a few seconds apart
        mean nothing moved on screen at all — the same "real progress" test
        ``ask()``'s busy-wait uses (chrome change OR pane.log growth)."""
        return _chrome(self._capture()), self._pane_log_size()

    def request_abort(self) -> None:
        """Ask every ask() that is ALREADY running on this session to give
        up at its next poll and release the pane lock (the owner's /reset).

        Without this the lock is held by a turn nobody is waiting for any
        more: an interrupted turn has no closing marker, so ask() sits out
        its whole no-reply ceiling (minutes), and a busy-wait on a pane that
        only LOOKS busy sits out the whole busy-wait budget. The stamp is a
        wall-clock time in a file, so it reaches an ask() in another process
        too, and a turn that starts after it is not affected."""
        tmp = self._abort_stamp.with_name(self._abort_stamp.name + ".tmp")
        try:
            tmp.write_text(f"{time.time():.6f}\n")
            os.replace(tmp, self._abort_stamp)
        except OSError:
            logger.warning("could not write the abort stamp", exc_info=True)

    def _abort_requested(self, since: float) -> bool:
        """True iff /reset asked turns started at ``since`` (wall clock) to
        stop. Missing or unreadable stamp ⇒ False."""
        try:
            return float(self._abort_stamp.read_text().strip()) >= since
        except (OSError, ValueError):
            return False

    def _aborted(self, log_id: str) -> AskResult:
        logger.warning("ask %s cut short by /reset — releasing the pane", log_id)
        self._inflight.unlink(missing_ok=True)
        return AskResult("reset", detail="turn cut short by /reset")

    def is_steerable_turn(self) -> bool:
        """True iff the in-flight turn may receive steering input.

        Maintenance turns (nightly pipeline, doctor canary, /process) tag
        themselves with a ``maint-`` request_id — user text steered into
        them would contaminate the background prompt and never get its own
        answer. A held lock WITHOUT an inflight record is startup/recovery/
        control — also not steerable.
        """
        if not self.is_turn_active():
            return False
        try:
            first = self._inflight.read_text().splitlines()[0]
        except (FileNotFoundError, IndexError):
            return False
        return not first.startswith(MAINT_PREFIX)

    def send_control(self, text: str) -> None:
        """Type a client-side Claude Code command verbatim, fire-and-forget.

        Control commands (/clear, /model, …) produce no model turn and thus
        no marker pair — there is nothing to extract, so don't wait.

        B1 fix (Fable audit fix round, 2026-08-22): `/clear` is special-
        cased here, not just in ``clear()``. The bot's actual `/clear`
        command (`bot/handlers/chat.py`'s `_CONTROL` set →
        `ChatSessionManager.send_control` → this method) and
        `cron_runner.py`'s post-job `/clear` BOTH call this method
        directly — neither ever goes through ``clear()`` — so the old code
        (resync logic living only in ``clear()``) meant the pinned
        `runtime_dir/session_id` went stale on every REAL `/clear` a user or
        cron job actually sent, and only stayed correct for the `/new` and
        `/compact` paths (which do call ``clear()``). See ``clear()``'s
        docstring for why the resync is necessary at all. Detecting on the
        literal command text (not a separate bool flag) is what guarantees
        this fires for every caller of this method, present and future,
        without relying on each call site to remember to opt in.
        """
        is_clear = text.strip() == "/clear"
        project_dir = self._transcript_project_dir()
        before: set[Path] = set()
        if is_clear:
            try:
                before = set(project_dir.glob("*.jsonl"))
            except OSError:
                before = set()
        # The resync runs INSIDE the pane lock (2026-09-25 fix). Released
        # between the keystroke and the resync, the lock let a queued chat
        # turn slip in, pin the OLD session id as its reply source, and then
        # see the resync as "session id changed mid-turn" — it dropped its
        # reply tail and sat out the full hour ceiling holding the lock, so
        # every later message hung behind it (seen live right after /new).
        # Holding the lock until the new id is pinned costs at most the
        # resync's own ~10 s window.
        with self._locked() as got:
            if got:
                self._send_text(text)
                self._send_enter()
            if is_clear and self._read_session_id() is not None:
                self._resync_session_id_after_clear(before)

    def clear(self) -> None:
        """Manual recovery only (durable-state-first: no scheduled clear).

        R1 caveat, VERIFIED LIVE 2026-08-22 (isolated throwaway session, NOT
        the brain — see the change contract for the transcript): `/clear`
        makes the CLI start an entirely NEW internal session id (a new
        `*.jsonl` file) — there is no flag or behavior that makes it keep
        the one this process was launched with via --session-id. Left
        unhandled, the pinned runtime_dir/session_id would go stale the
        instant this runs, silently breaking every transcript-dependent
        feature (the reply source ask() reads, R3's context check,
        marker_compliance.py) with no error until someone notices the
        transcript stopped growing — so this resyncs the pin immediately
        after. An ask() already waiting when this fires notices the id
        changed under it and says so (see its session-id check).

        Thin wrapper over ``send_control("/clear")`` — the resync logic
        lives THERE now (B1 fix, 2026-08-22) precisely so every caller that
        sends `/clear`, not just this method's own callers (`/new`,
        `/compact`), gets the resync for free.
        """
        self.send_control("/clear")

    # ── sending ──────────────────────────────────────────────────────

    def _send_text(self, text: str) -> None:
        # Stream the payload to `load-buffer -` over stdin; passing it as an
        # argv element trips tmux's "set-buffer: command too long" on long
        # prompts and the text is silently dropped (session then stalls).
        if not text:
            return  # 0 bytes ⇒ no buffer ⇒ paste-buffer would fail `no buffer`
        buf = f"dbrain_{uuid.uuid4().hex[:6]}"
        self._tmux("load-buffer", "-b", buf, "-", input_text=text)
        self._tmux("paste-buffer", "-t", self._target, "-b", buf, "-d")
        self._sleep(self._paste_settle)

    def _send_prompt(self, prompt: str, rid: str, *, wrap: bool = True) -> None:
        # Markers are written INLINE (mid-sentence) so the input echo never
        # forms a line-anchored pair; only the model's answer does.
        if not wrap:
            self._send_text(prompt)
            self._send_enter()
            self._confirm_submitted()
            return
        payload = (
            f"{prompt}\n\n"
            f"When done, wrap your ENTIRE reply between a line containing only "
            f"<<<R:{rid}>>> and a line containing only <<<E:{rid}>>>. "
            f"The reply is delivered ONLY after the <<<E:{rid}>>> line — end "
            f"with it."
        )
        self._send_text(payload)
        self._send_enter()
        self._confirm_submitted(f"<<<E:{rid}>>>")

    def _confirm_submitted(self, marker: str | None = None) -> None:
        """Press Enter again if the first one was lost.

        Reproduced live on Claude Code 2.1.278 (isolated tmux server): a
        31-line paste, 0.3-0.5 s, Enter — and the prompt just sat in the
        input box; the next prompt was then pasted on top of it (seen live
        once: five glued, unsent prompts). A second Enter sent it. An extra
        Enter on an already-sent, empty box is a no-op
        both while idle and during a running turn (live: no empty user
        record in the transcript, the turn not disturbed).

        The box is judged by input_box_text() only — the transcript echo of
        a sent prompt carries the same marker. Still unsent = our marker or
        a collapsed "[Pasted text …]" is visible in it. At most
        _ENTER_RETRIES extra Enters, each logged."""
        for attempt in range(_ENTER_RETRIES + 1):
            self._sleep(self._submit_settle)
            box = input_box_text(self._capture())
            if box is None or not (
                _PASTED_TEXT in box or (marker is not None and marker in box)
            ):
                return
            if attempt == _ENTER_RETRIES:
                logger.error(
                    "session %s: prompt still unsent in the input box after "
                    "%d extra Enters",
                    self.session_name,
                    _ENTER_RETRIES,
                )
                return
            logger.warning(
                "session %s: Enter lost — prompt still in the input box, "
                "pressing Enter again (%d/%d)",
                self.session_name,
                attempt + 1,
                _ENTER_RETRIES,
            )
            self._send_enter()

    # ── the reply source ──────────────────────────────

    def _open_reply_tail(
        self, rid: str, log_id: str
    ) -> tuple[ReplyTail | None, str | None]:
        """A :class:`ReplyTail` anchored at the CURRENT end of this session's
        pinned transcript — the source ask() reads the reply from — together
        with the session id it belongs to.

        ``(None, None)`` (with a loud log line) if there is no pinned session
        id or the tail cannot be opened at all. That turn has no transcript to
        read and falls back to the pane, which is strictly a degraded mode —
        see the fallback in ask(). Never raises."""
        try:
            sid = self._read_session_id()
            if sid is None:
                logger.error(
                    "no pinned session id — cannot read the reply for %s from "
                    "the transcript; falling back to reading the pane, which "
                    "loses any reply longer than the capture window",
                    log_id,
                )
                return None, None
            return ReplyTail(transcript_path(self.work_dir, sid), rid), sid
        except Exception:  # noqa: BLE001 — never take a live turn down
            logger.error(
                "could not open the transcript tail for %s", log_id, exc_info=True
            )
            return None, None

    def _poll_reply(self, tail: ReplyTail | None, log_id: str):
        """This turn's reply as the transcript currently has it, or ``None``.

        Wrapped so a transcript-shape surprise (the JSONL schema is not a
        published API) degrades to "nothing found yet" — the caller keeps
        polling and ends on its existing ceiling — never to an exception out
        of ask()."""
        if tail is None:
            return None
        try:
            return tail.poll()
        except Exception:  # noqa: BLE001 — never take a live turn down
            logger.error(
                "transcript poll failed for %s — treating as 'no reply yet'",
                log_id,
                exc_info=True,
            )
            return None

    # ── ask ──────────────────────────────────────────────────────────

    def ask(
        self,
        prompt: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        request_id: str | None = None,
        wrap: bool = True,
    ) -> AskResult:
        """Send a prompt, return the model's reply (or a non-ok status).

        Lock-guarded so concurrent callers serialize on the pane. Never
        raises and never blocks forever: ensure failure → error, rate-limit /
        logged-out short-circuit, a pane with no liveness signal (stall)
        interrupts the turn and releases the lock, and a hard timeout returns
        a timeout status. request_id is for logging only; the marker rid is
        always freshly generated to avoid stale answers.

        wrap=True (default) appends the marker instruction and extracts the
        reply between the marker pair — the reliable path for model turns.
        wrap=False types the prompt verbatim; completion is the pane sitting
        idle for two consecutive polls and the reply is the chrome-stripped
        pane text (best effort, may include the input echo).
        """
        rid = self._rid_factory()
        log_id = request_id or rid
        with self._locked() as got:
            if not got:  # only happens with non-blocking; blocking=True here
                return AskResult("error", detail="could not acquire pane lock")
            # Claim the turn IMMEDIATELY: a stale inflight from a timed-out
            # earlier turn must not misrepresent this holder to the steering
            # gate while _ensure_locked() spends up to startup_timeout. The
            # claim is a MAINT-prefixed placeholder, not our real id: nothing
            # of ours is in the pane yet, and _ensure_locked() may be driving
            # it through the "← N agents" list (_return_to_main) or starting
            # a fresh session. With the real id here is_steerable_turn() said
            # True, and an owner message steered in at that moment landed in
            # the list's "describe a task for a new session" box — spawning a
            # background session — or in a half-started TUI (review
            # 2026-09-19). The real id is written only right before the
            # prompt is typed, same as after the busy-wait below.
            self._inflight.write_text(
                f"{MAINT_PREFIX}pending-{log_id}\n{self._clock()}\n"
            )
            # Wall clock, for request_abort (a stamp from another process).
            turn_started = time.time()
            try:
                reused_cap = self._ensure_locked()
            except Exception as exc:  # noqa: BLE001 — must never escape ask()
                logger.error("ensure_session failed for %s: %s", log_id, exc)
                self._inflight.unlink(missing_ok=True)
                return AskResult("error", detail=f"session start failed: {exc}")

            # ONE deadline for the whole call, computed BEFORE the busy-wait
            # below: a separate stall_timeout budget for the busy-wait used
            # to be added ON TOP of timeout (900 + 3600), stretching the
            # worst-case turn well past the documented hour ceiling (review
            # 2026-08-20). The busy-wait now draws down the SAME budget the
            # rest of the turn uses.
            deadline = self._clock() + timeout

            # The view check in _ensure_locked() just captured this pane under
            # the same lock; reuse that frame rather than taking a second one.
            pre_cap = reused_cap if reused_cap is not None else self._capture()
            pre = classify_state(pre_cap)
            if pre == PaneState.RATE_LIMITED and self._confirmed_rate_limited(
                pre_cap, log_id
            ):
                logger.warning(
                    "rate-limited (pre-send) for %s — refusing to send", log_id
                )
                self._inflight.unlink(missing_ok=True)
                return AskResult("rate_limited")
            if pre == PaneState.LOGGED_OUT:
                self._inflight.unlink(missing_ok=True)
                return AskResult("logged_out")
            if not is_main_turn_active(pre_cap) and is_working(pre_cap):
                # Defect B (-08-21): background-agent
                # list rows alone make bare is_working() True forever (their
                # elapsed counter ticks every second), which used to refuse
                # to type at all — the busy-wait below would burn its whole
                # DEFAULT_BUSY_WAIT_BUDGET and hand back "pane still busy"
                # even though the MAIN turn is demonstrably idle. Logged so
                # this path is visible in the journal rather than silently
                # absent (§6 of the plan).
                logger.info(
                    "pre-send: background-agent rows present but main turn "
                    "idle for %s — sending anyway instead of waiting",
                    log_id,
                )
            if is_main_turn_active(pre_cap):
                # A PREVIOUS ask() can have released the pane lock on a
                # stall (2026-08-20 fix) while its own turn kept running —
                # is_turn_active() is lock-based, so acquiring the lock here
                # does not mean the pane itself is idle. Typing a new prompt
                # over a live turn interleaves both turns' text in one pane
                # and corrupts both (found in review 2026-08-20). Wait for
                # the leftover turn to clear rather than type over it.
                #
                # Uses the same "progress OR pane.log growth" pair the main
                # stall loop below uses — NOT bare is_working() — so a
                # previous turn that is itself frozen (dead subagent,
                # D-state process) is not waited on for the full remaining
                # budget just because a static "working" signature happens
                # to still be on screen (review 2026-08-20).
                #
                # inflight keeps the MAINT-prefixed placeholder written at
                # claim time for the whole wait: we are not typing anything
                # yet, and the pane belongs to that OTHER, still-running
                # turn — our real (non-maint) log_id here, before the pane
                # is confirmed free, made is_steerable_turn() return True
                # for that other turn, so an input arriving during the wait
                # would be steered straight into it, interleaving both
                # turns' text in one pane (found in review 2026-08-20).
                cap = pre_cap

                # B3 fast path: the watchdog
                # may already have evidence, from its own independent
                # polling, that this is a live/progressing unattended long
                # run — see long_run.py's module docstring for the marker
                # contract. Confirm it with a SHORT poll window against the
                # live pane (evidence-based, never a blind trust of the
                # marker file) instead of paying the full busy-wait budget
                # to re-discover the same fact. A missing/stale marker falls
                # straight through to today's unchanged behavior below.
                long_active, long_elapsed = long_run.is_active(
                    self.runtime_dir,
                    now=time.time(),
                    stale_after=self._long_run_stale_after,
                )
                if long_active:
                    confirm_deadline = self._clock() + max(
                        3.0, self._poll_interval * _LONG_RUN_CONFIRM_POLLS
                    )
                    confirm_last_log_size = self._pane_log_size()
                    confirmed = False
                    while self._clock() < confirm_deadline:
                        self._sleep(self._poll_interval)
                        new_cap = self._capture()
                        log_size = self._pane_log_size()
                        # F1 fix (blind-review round): a REAL chrome change or
                        # pane.log growth only — deliberately NOT
                        # is_working_progressing(), which exempts the static
                        # "esc to interrupt" hint forever (see
                        # _RECENT_PROGRESS_POLLS' docstring above). This is a
                        # confirmation of genuine liveness, not a "should I
                        # keep waiting" check.
                        if (
                            _chrome(new_cap) != _chrome(cap)
                            or log_size > confirm_last_log_size
                        ):
                            confirmed = True
                        confirm_last_log_size = log_size
                        cap = new_cap
                        if confirmed:
                            break
                    if confirmed and is_main_turn_active(cap):
                        logger.info(
                            "pane busy with a live, progressing turn "
                            "(long-run marker confirmed, ~%.0fs elapsed) — "
                            "not typing over it for %s",
                            long_elapsed,
                            log_id,
                        )
                        self._inflight.unlink(missing_ok=True)
                        return AskResult(
                            "busy_active",
                            detail="pane busy with a live, progressing turn",
                            busy_seconds=long_elapsed,
                        )
                    # Not confirmed against the live pane (or the leftover
                    # turn actually cleared during the confirm window) —
                    # fall through to the full busy-wait below. A genuinely
                    # wedged pane must still consume the full budget and end
                    # as plain "busy" (the B3 regression guard).

                busy_wait_start = self._clock()
                busy_wait_deadline = self._clock() + self._busy_wait_budget
                busy_last_active = self._clock()
                busy_last_log_size = self._pane_log_size()
                # F1 fix (blind-review round): TWO SEPARATE progress
                # trackers, deliberately not sharing one predicate.
                #
                # busy_last_active (unchanged, still is_working_progressing()
                # OR log growth) answers "should I keep waiting" — the
                # static "esc to interrupt" hint counting as liveness there
                # is correct: it exists so this loop doesn't abandon a
                # silently-alive turn just because the stall_timeout window
                # has no OTHER signal to look at.
                #
                # last_real_progress_ts answers a different question, "is
                # this pane ALIVE right now" — the one the busy/busy_active
                # split needs — and must NOT get that exemption: a pane
                # whose only signal is a static hint that never actually
                # changes is exactly the wedge class B3 (2026-08-22) exists
                # to catch. It only advances on a REAL chrome change or
                # pane.log growth, and classification at give-up time
                # requires that observation to be RECENT (within
                # _RECENT_PROGRESS_POLLS polls), not merely "ever seen" —
                # a latch that survives the rest of a frozen wait would
                # silently reopen the same hole (see this constant's
                # docstring above).
                last_real_progress_ts = busy_wait_start
                while self._clock() < deadline and is_main_turn_active(cap):
                    if self._abort_requested(turn_started):
                        return self._aborted(log_id)
                    if self._clock() >= busy_wait_deadline:
                        break
                    if self._clock() - busy_last_active > self._stall_timeout:
                        break
                    self._sleep(self._poll_interval)
                    new_cap = self._capture()
                    log_size = self._pane_log_size()
                    # last_real_progress_ts is already the
                    # exact "real signal" tracker this needs (chrome change OR
                    # log growth, no static-hint exemption), so the trust
                    # window rides on it — nothing new to track here. Read
                    # BEFORE this iteration updates it, so the decision is
                    # made on evidence that already existed.
                    trust_static = (
                        self._clock() - last_real_progress_ts
                    ) < STATIC_TRUST_WINDOW
                    if (
                        is_working_progressing(new_cap, cap, trust_static=trust_static)
                        or log_size > busy_last_log_size
                    ):
                        busy_last_active = self._clock()
                    if (
                        _chrome(new_cap) != _chrome(cap)
                        or log_size > busy_last_log_size
                    ):
                        last_real_progress_ts = self._clock()
                    busy_last_log_size = log_size
                    cap = new_cap
                recent_window = self._poll_interval * _RECENT_PROGRESS_POLLS
                saw_progress = (self._clock() - last_real_progress_ts) <= recent_window
                if is_main_turn_active(cap):
                    self._inflight.unlink(missing_ok=True)
                    busy_seconds = self._clock() - busy_wait_start
                    if saw_progress:
                        # Progress-aware classification: the pane made a
                        # REAL, RECENT change — a live turn, not a latched
                        # static signature.
                        # Distinct status ("busy_active") so ask_health
                        # treats this as neutral rather than a delivery
                        # failure — see that module's docstring.
                        logger.info(
                            "pane busy with a live, progressing turn — not "
                            "typing over it for %s",
                            log_id,
                        )
                        return AskResult(
                            "busy_active",
                            detail="pane busy with a live, progressing turn",
                            busy_seconds=busy_seconds,
                        )
                    logger.error(
                        "pane still busy with a previous turn after waiting — "
                        "refusing to type over it for %s",
                        log_id,
                    )
                    # Separate 'error'-CLASS but honest status ("2026-08-22
                    # busy-panel UX" finding): the panel is legitimately busy
                    # with someone else's long-running turn — chat_session.py
                    # maps this to a distinct, friendlier MESSAGE than
                    # "❌ Ошибка сессии", which misled the owner into thinking
                    # something had crashed when the session was
                    # simply still working. B3 fix (2026-08-22): "busy" DOES
                    # count in ask_health.FAILURE_STATUSES now — the original
                    # exclusion assumed a busy pane is never evidence the
                    # delivery path is broken, but is_working()/
                    # is_main_turn_active() can key on a persistent footer
                    # signature that never changes for a genuinely wedged
                    # pane too, so excluding "busy" silently disabled
                    # delivery_guard's restart backstop for that case. Only
                    # the wording stays distinct; the health accounting does
                    # not — this NO-PROGRESS branch stays byte-identical to
                    # before this change (see busy_seconds' additive-only
                    # field docstring).
                    return AskResult(
                        "busy",
                        detail="pane still busy with a previous turn",
                        busy_seconds=busy_seconds,
                    )
                pre_cap = cap

            # The pane is confirmed ours and free: take on our real identity
            # (steerable from here on) right before typing anything into it.
            self._inflight.write_text(f"{log_id}\n{self._clock()}\n")

            # THE reply source for this turn, anchored
            # BEFORE the prompt is typed so nothing the model writes in answer
            # to it can land ahead of the anchor — the rid is fresh, so
            # records left over from an earlier turn cannot match it either.
            # The pane is read from here on for STATE ONLY; the reply text
            # comes from the transcript and nowhere else.
            # The transcript the tail follows belongs to THIS session id. A
            # `/clear` (or any re-key) mid-wait starts a brand-new JSONL file,
            # leaving the tail pointed at one that has stopped growing — see
            # the session-id check inside the loop.
            reply_tail, sid_at_send = (
                self._open_reply_tail(rid, log_id) if wrap else (None, None)
            )
            # Fixed at send time on purpose: a tail DROPPED mid-turn (the
            # session was re-keyed under us) must not silently switch this
            # turn over to reading the pane — that is a different
            # conversation's screen by then. Only a turn that never had a
            # transcript to read at all may fall back.
            never_had_transcript = wrap and reply_tail is None

            self._send_prompt(prompt, rid, wrap=wrap)

            last_active = self._clock()
            last_cap = pre_cap
            last_log_size = self._pane_log_size()
            # last REAL movement of the pane (chrome change
            # OR pane.log growth), with no static-hint exemption — the clock
            # the STATIC_TRUST_WINDOW runs on for this loop. Separate from
            # last_active above, which by design DOES take the static hint as
            # liveness (that exemption is what keeps a quiet turn alive).
            last_chrome_change_ts = last_active
            # Fixed reference for F5's latch fix below — unlike last_cap
            # (updated every iteration for stall detection), this stays the
            # exact frame observed at send time.
            send_time_cap = pre_cap
            main_turn_seen_active_or_changed = False
            # F2 tracking (R2c): has a <<<R:rid>>> opening marker EVER been
            # seen in the transcript this turn? Distinguishes "the model
            # started answering but never closed the pair" from "no markers
            # at all" for the ceiling's honest message. Unlike the pane, the
            # transcript cannot lose a marker it once had, so this is now a
            # statement about the MODEL, not about the capture window.
            ever_saw_r_marker = False
            idle_streak = 0
            rate_limited_streak = 0
            # ── salvage/ceiling tracking state ──────────
            # send_time: fixed anchor for "how long have we waited" logging
            # below — unlike last_active/last_main_turn_active, this never
            # moves.
            send_time = last_active
            # last_main_turn_active: the last poll on which the MAIN turn
            # (not a background-agent row) was observed active. Initialized
            # to the moment the prompt was sent — same baseline as
            # last_active above.
            last_main_turn_active = last_active
            # salvage_region / salvage_stable_since: track content stability
            # of the SALVAGE REGION ONLY (never the whole pane, which changes
            # every second from ticking background-agent rows — that would
            # defeat any whole-pane stability check entirely).
            salvage_region: str | None = None
            salvage_stable_since = last_active
            main_turn_finished_logged = False
            # Latched separately from main_turn_finished_logged: fires once,
            # the first poll the salvage WINDOW is open (last_main_turn_active
            # is at least salvage_stable seconds old) but the salvage
            # condition as a whole still refuses — closing the "salvage
            # refusal path logs nothing at all" gap (checklist + 7,
            # 2026-08-22 plan): before this, the only way to know WHY a given
            # turn missed salvage and rode the ceiling instead was a fresh
            # forensic pass over pane.log.
            salvage_blocked_logged = False
            while self._clock() < deadline:
                if self._abort_requested(turn_started):
                    return self._aborted(log_id)
                cap = self._capture()
                transcript_reply = None
                if wrap:
                    # `/clear` (or any other re-key) mid-wait starts a NEW
                    # transcript file: the tail we hold stops growing and can
                    # never carry this turn's reply. Stop reading it — and say
                    # so — rather than silently waiting out the ceiling on a
                    # dead file. The turn itself is left alone (it may still
                    # be running); it ends on the ceiling below, rid unhandled
                    # for the orphan poller, exactly as any other lost-reply
                    # turn does.
                    if reply_tail is not None:
                        sid_now = self._read_session_id()
                        # A read that fails transiently returns None — that is
                        # "unknown", not "re-keyed", and must not cost this
                        # turn its reply source. Only a DIFFERENT, readable id
                        # means the file we follow is no longer this session's.
                        if sid_now is not None and sid_now != sid_at_send:
                            logger.error(
                                "session id changed mid-turn for %s (%s → %s) "
                                "— the transcript this turn was being read "
                                "from is gone; no reply can be recovered here",
                                log_id,
                                sid_at_send,
                                sid_now,
                            )
                            reply_tail = None
                    transcript_reply = self._poll_reply(reply_tail, log_id)
                    if reply_tail is not None and reply_tail.saw_open:
                        # Deliberately the MARKER, not the reply: a turn that
                        # opened the pair and wrote nothing after it did start
                        # answering, and must not be reported as "no markers
                        # ever appeared".
                        ever_saw_r_marker = True
                # COMPLETION IS CHECKED FIRST, before any fault state. Our own
                # finished answer outranks whatever else is on the pane: with
                # the checks the other way round, a reply that merely
                # mentioned a limit made ask() return "rate_limited" and throw
                # the finished answer away — the user got "⏳ Лимит подписки
                # исчерпан" and then the real answer seconds later from the
                # orphan poller. That is the 2026-08-20 double-message bug.
                #
                # "Complete" is now the CLOSING MARKER IN THE TRANSCRIPT, not
                # on the pane: a reply longer than the
                # capture window pushes its own opening marker out of frame,
                # so the pane can show a finished answer that no parser can
                # recognise. An unclosed span is deliberately NOT delivered
                # here — half a reply is worse than a late one; it is what the
                # salvage/ceiling rules below judge, once the pane says the
                # turn is over.
                #
                # The pane is read for the reply in ONE case only: this
                # session has no transcript at all (no pinned session id, or
                # the tail could not be opened). That is degraded mode, not a
                # second source — the two are mutually exclusive by
                # construction, so a turn still takes its reply from exactly
                # one place — and it beats timing out on every single turn
                # against an otherwise healthy pane, which is what a lost pin
                # would otherwise cost now that nothing else reads the screen.
                pane_fallback = never_had_transcript and is_complete(cap, rid)
                if pane_fallback:
                    logger.warning(
                        "delivering %s from the PANE: this session has no "
                        "readable transcript, so any reply longer than the "
                        "capture window would have been lost",
                        log_id,
                    )
                if pane_fallback or (
                    wrap and transcript_reply is not None and transcript_reply.closed
                ):
                    reply = (
                        extract_reply(cap, rid)
                        if pane_fallback
                        else transcript_reply.body
                    )
                    self._inflight.unlink(missing_ok=True)
                    migrated = self._ensure_migrated(cap)
                    if not migrated:
                        # Any OTHER unhandled pair still on screen (a
                        # background subagent's orphan this very turn
                        # happened to finish ahead of) is about to fall below
                        # the watermark the instant our rid is marked handled
                        # below — queue it so pop_orphan_replies() can still
                        # find it (review 2026-08-20; see _pending_orphans
                        # field docstring). Skipped when migration just ran
                        # THIS call: handled_rids already reflects everything
                        # on screen, so "older" here would be the whole
                        # pre-existing scrollback, not a genuine orphan.
                        #
                        # Must use find_pending_replies (watermark-aware),
                        # not find_unhandled_replies (raw "anything not in
                        # handled"): pending_orphans is handed out by
                        # pop_orphan_replies() UNCONDITIONALLY, ignoring the
                        # watermark itself (see that method's comment) — so
                        # anything queued here bypasses the watermark for
                        # good. Queuing truly old, already-delivered pairs
                        # that a pane reflow merely brought back into the
                        # capture window would resend them a second time
                        # (round-2 finding, 2026-08-20).
                        handled_now = set(self._read_handled())
                        older = [
                            r
                            for r, _ in find_pending_replies(cap, handled_now)
                            if r != rid
                        ]
                        self._queue_pending_orphans(older)
                    # Record as handled so the orphan poller never re-delivers
                    # this rid once no ask() is waiting on it. A write
                    # failure is logged (ERROR) inside _mark_handled; the
                    # reply still goes to the user regardless — this is the
                    # only delivery channel for it, so suppressing it over a
                    # filesystem hiccup would be strictly worse than the rare
                    # risk of a later duplicate.
                    self._mark_handled(rid)
                    return AskResult("ok", reply=reply)
                # R5 (Fable audit / F4): strip OUR OWN unclosed span before
                # classifying pane state. An in-progress or marker-dropped
                # reply that happens to quote rate-limit-looking text
                # otherwise sits in the chrome window (strip_reply_bodies
                # only ever removes CLOSED pairs) and can make a real
                # RATE_LIMITED classification poll-resistant — the frame is
                # static, so _RATE_LIMIT_CONFIRM_POLLS never clears it. The
                # completion check above is unaffected (runs on raw `cap`).
                # Our own (unclosed) reply text removed from the frame — see
                # R5 below. Computed once: the salvage gate needs the same
                # cleaned frame to ask "is the CLI still working?" without the
                # model's own prose answering for it.
                clean_cap = strip_open_reply_body(cap, rid) if wrap else cap
                state = classify_state(clean_cap)
                if state == PaneState.RATE_LIMITED:
                    # Require the signature to persist across several polls
                    # before trusting it — see _RATE_LIMIT_CONFIRM_POLLS.
                    rate_limited_streak += 1
                    if rate_limited_streak >= _RATE_LIMIT_CONFIRM_POLLS:
                        logger.warning(
                            "rate-limited (in-loop-streak) for %s after %d "
                            "consecutive polls",
                            log_id,
                            rate_limited_streak,
                        )
                        self._inflight.unlink(missing_ok=True)
                        return AskResult("rate_limited")
                else:
                    rate_limited_streak = 0
                auth_failed = wrap and turn_auth_error(cap, rid)
                if state == PaneState.LOGGED_OUT or auth_failed:
                    self._inflight.unlink(missing_ok=True)
                    return AskResult("logged_out")
                if not wrap and is_idle(cap):
                    idle_streak += 1
                    if idle_streak >= 2:
                        self._inflight.unlink(missing_ok=True)
                        return AskResult("ok", reply=strip_chrome(cap))
                else:
                    idle_streak = 0

                # The periodic "How is Claude doing?" survey pollutes the
                # chrome and once made the stall detector interrupt a live
                # turn. Dismiss it (0) and never count it as a stall.
                if has_survey_prompt(cap):
                    self._tmux("send-keys", "-t", self._target, "0")
                    last_active = self._clock()
                    self._sleep(self._poll_interval)
                    continue

                # ── salvage / no-main-turn ceiling ───────
                # Scoped to wrap=True: there is no closing-marker concept for
                # wrap=False turns (those complete on two idle polls, above),
                # so neither salvage nor the ceiling applies to them.
                #
                # The candidate is the transcript's UNCLOSED span for this rid
                # — the model opened the pair and wrote an
                # answer but never emitted `<<<E:rid>>>` (~3% of turns,
                #). Two rules decide whether to deliver it:
                # the PANE must say the main turn is over, and the BODY must
                # have stopped growing. Both are needed — the pane alone can
                # false-negative on a live turn, and a body that is still
                # growing is a half-written answer.
                #
                # This replaced extract_open_reply()'s pane scraping, and with
                # it two known limitations of that approach: salvage needed a
                # recognisable TUI boundary line after the reply to stop at
                # (absent one, it silently never fired), and its boundary
                # signatures matched as substrings, so a reply whose own prose
                # contained "Worked for" or "❯" could be truncated or refused.
                # The transcript body has no chrome in it to confuse either
                # way, which is also why the old `_WORKING_RE` conjunct on the
                # candidate text is gone: it existed to catch pane chrome
                # swept up into the region, and there is no pane chrome here.
                if wrap:
                    if is_main_turn_active(cap):
                        last_main_turn_active = self._clock()
                        main_turn_seen_active_or_changed = True
                    elif not main_turn_seen_active_or_changed and cap != send_time_cap:
                        main_turn_seen_active_or_changed = True
                    if (
                        # F5 (Fable audit): only trust main_turn_finished()
                        # once we've confirmed we're looking at THIS turn's
                        # frame — either the main spinner was seen active at
                        # least once, or the pane has visibly changed since
                        # send. Without this gate, the FIRST poll (0.0s after
                        # send) can still be showing the PREVIOUS, already-
                        # finished turn's frame, and a false "main turn
                        # finished ... 0.0s after send" INFO line was logged
                        # on every turn regardless of whether anything had
                        # actually happened yet.
                        main_turn_seen_active_or_changed
                        and not main_turn_finished_logged
                        and main_turn_finished(cap)
                    ):
                        # Loud on purpose: this incident's defining property
                        # was total silence in the journal for 22 minutes —
                        # log the exact moment the main turn is first seen
                        # finished, not just silence.
                        main_turn_finished_logged = True
                        logger.info(
                            "main turn finished (no closing marker yet) for "
                            "%s, %.1fs after send",
                            log_id,
                            self._clock() - send_time,
                        )

                    # The unclosed span as of THIS poll (the closed case
                    # returned far above). Recomputed every poll and never
                    # trusted from the cached tracking value below, so a body
                    # that is still growing keeps resetting its own stability
                    # clock instead of being delivered half-written.
                    fresh_open = (
                        transcript_reply.body
                        if transcript_reply is not None and not transcript_reply.closed
                        else None
                    )
                    if fresh_open != salvage_region:
                        salvage_region = fresh_open
                        salvage_stable_since = self._clock()

                    # A working signature in the conversation area, not
                    # just the main-turn spinner. This is the replacement for
                    # the `_WORKING_RE` conjunct the pane-scraping salvage
                    # applied to its scraped region, and it exists for the
                    # same reason: is_main_turn_active() can false-negative on
                    # a genuinely live turn (the reviewer-demonstrated
                    # non-paren spinner "✢ Razzle-dazzling…  44s · ↓1.8k
                    # tokens"). Without it, a turn that opened the pair, wrote
                    # a preamble and then spent minutes inside one tool call
                    # has a body that is BYTE-IDENTICAL for the whole salvage
                    # window — and would be delivered as a fragment, with its
                    # rid marked handled, permanently blocking the real answer
                    # that arrives later. Scoped to the conversation area
                    # (main_area_working) for the reasons in that function's
                    # docstring. At the full ceiling below this no longer
                    # applies (R2b).
                    # Deliberately the RAW frame, NOT `clean_cap`: stripping
                    # our own open span also strips a non-paren spinner line
                    # rendered right after the reply text, because
                    # _is_boundary_line does not recognise that shape and so
                    # sweeps it INTO the span (its docstring calls this out).
                    # Cleaning first would make this net dead code for
                    # exactly the shape it exists to catch — verified: the
                    # Razzle-dazzling fixture salvages early with `clean_cap`
                    # and correctly refuses with the raw frame. The model's
                    # own prose is therefore inside what this reads, which is
                    # why that scan has to be fail-safe about it.
                    pane_still_working = wrap and main_area_working(cap)
                    if (
                        main_turn_finished(cap)
                        and not pane_still_working
                        and self._clock() - last_main_turn_active
                        >= self._salvage_stable
                        and fresh_open is not None
                        and self._clock() - salvage_stable_since >= self._salvage_stable
                    ):
                        # Mirror the EXISTING normal-completion path exactly
                        # (release inflight → migrate/queue older orphans →
                        # mark THIS rid handled) so a later-appearing E marker
                        # (a delayed repaint) can never re-deliver this same
                        # text a second time.
                        elapsed = self._clock() - send_time
                        self._inflight.unlink(missing_ok=True)
                        migrated = self._ensure_migrated(cap)
                        if not migrated:
                            handled_now = set(self._read_handled())
                            older = [
                                r
                                for r, _ in find_pending_replies(cap, handled_now)
                                if r != rid
                            ]
                            self._queue_pending_orphans(older)
                        self._mark_handled(rid)
                        logger.warning(
                            "salvaged reply for %s: missing closing marker, "
                            "waited %.1fs, salvaged body length=%d chars",
                            log_id,
                            elapsed,
                            len(fresh_open),
                        )
                        return AskResult("ok", reply=fresh_open, salvaged=True)

                    region_desc = (
                        "None" if fresh_open is None else f"{len(fresh_open)} chars"
                    )
                    if (
                        not salvage_blocked_logged
                        and self._clock() - last_main_turn_active
                        >= self._salvage_stable
                    ):
                        salvage_blocked_logged = True
                        logger.info(
                            "salvage window open for %s but not taken: "
                            "main_turn_finished=%s pane_still_working=%s "
                            "transcript_region=%s region_stable_for=%.1fs "
                            "since_main_turn_active=%.1fs",
                            log_id,
                            main_turn_finished(cap),
                            pane_still_working,
                            region_desc,
                            self._clock() - salvage_stable_since,
                            self._clock() - last_main_turn_active,
                        )

                    if (
                        self._clock() - last_main_turn_active
                        >= self._no_main_turn_ceiling
                    ):
                        # R2b (Fable audit): at the FULL ceiling, deliver the
                        # open span even though the faster salvage path above
                        # refused it (its body never stayed still for a whole
                        # salvage window). By the time a full
                        # no_main_turn_ceiling has passed with no confirmed
                        # main-turn activity AT ALL, a possibly-truncated
                        # reply is explicitly judged (per the audit) better
                        # than the alternative this branch used to guarantee:
                        # permanent loss, since a rid released unhandled here
                        # can only ever be recovered by a complete R/E pair
                        # later (F1) — never by this same open span. ACCEPTED
                        # RISK, stated plainly: if is_main_turn_active() is
                        # false-negativing on a STILL-RUNNING turn, this can
                        # salvage a truncated answer and permanently block the
                        # real, complete one via handled_rids. Marked
                        # `salvaged=True` either way so chat_session.py's
                        # existing warning notice covers it (no new
                        # user-facing wording needed).
                        if main_turn_finished(cap) and fresh_open is not None:
                            elapsed = self._clock() - send_time
                            self._inflight.unlink(missing_ok=True)
                            migrated = self._ensure_migrated(cap)
                            if not migrated:
                                handled_now = set(self._read_handled())
                                older = [
                                    r
                                    for r, _ in find_pending_replies(cap, handled_now)
                                    if r != rid
                                ]
                                self._queue_pending_orphans(older)
                            self._mark_handled(rid)
                            logger.warning(
                                "ceiling salvage for %s: no closing marker, no "
                                "confirmed main-turn activity for %.1fs "
                                "(waited %.1fs total), delivering possibly-"
                                "truncated transcript body (len=%d) rather "
                                "than a bare timeout",
                                log_id,
                                self._clock() - last_main_turn_active,
                                elapsed,
                                len(fresh_open),
                            )
                            return AskResult("ok", reply=fresh_open, salvaged=True)

                        # F2 class (Fable audit R2c): nothing to salvage —
                        # the model never opened a `<<<R:rid>>>` pair in the
                        # transcript at all, or this turn lost its transcript
                        # (no pinned session id, an unreadable file, a
                        # `/clear` mid-wait). Distinguish the two for an
                        # honest user-facing message: chat_session.py greps
                        # `detail` for "no reply markers ever appeared".
                        #
                        # Do NOT send Escape/Ctrl-C, do NOT interrupt: if
                        # is_main_turn_active ever false-negatives on a
                        # genuinely still-live turn, THIS ceiling path itself
                        # must cost a DELAYED reply, never a KILLED one. Do
                        # NOT mark the rid handled either — leaving it
                        # unhandled keeps the orphan-delivery route open: if
                        # the turn later actually completes with a proper
                        # pair, the NEXT ask() call's completion path queues
                        # it into pending_orphans via find_pending_replies
                        # and the watchdog delivers it later, with zero
                        # changes to the existing orphan machinery.
                        #
                        # HONEST CAVEAT (round-2 review, 2026-08-21): "cost is
                        # bounded at delayed" is true for THIS ceiling exit on
                        # its own, but incomplete — is_main_turn_active is the
                        # SAME predicate that also gates the PRE-SEND check
                        # above (Defect B fix). So once this ceiling releases
                        # the lock, the user's very next message can re-enter
                        # ask(), hit the SAME false negative on the pre-send
                        # gate, and type a new prompt into a still-live turn —
                        # interleaving two turns' text in one pane, a risk
                        # class this codebase has previously identified as
                        # dangerous (see the pre-send is_main_turn_active
                        # check above). This is an ACCEPTED, INTENTIONAL
                        # tradeoff: fixing Defect B (the false "busy"
                        # pre-send rejection) requires reusing this exact
                        # predicate. Not claiming otherwise here.
                        self._inflight.unlink(missing_ok=True)
                        logger.error(
                            "no closing marker and no main-turn activity "
                            "detected for %s (%.1fs since last active) — "
                            "lock released without interrupting, rid left "
                            "unhandled for the orphan poller: "
                            "main_turn_finished=%s pane_still_working=%s "
                            "transcript_region=%s region_stable_for=%.1fs "
                            "transcript_records_seen=%s ever_saw_r_marker=%s",
                            log_id,
                            self._clock() - last_main_turn_active,
                            main_turn_finished(cap),
                            pane_still_working,
                            region_desc,
                            self._clock() - salvage_stable_since,
                            reply_tail is not None and reply_tail.saw_records,
                            ever_saw_r_marker,
                        )
                        if reply_tail is not None and not reply_tail.saw_records:
                            # A tail that consumed NOTHING for a whole turn is
                            # following a file nobody writes — a stale pin, or
                            # a `claude` restarted inside the pane under
                            # another id. Nothing else in the journal says so,
                            # and left unnoticed it is a timeout on every
                            # single turn.
                            logger.error(
                                "the transcript this session is pinned to (%s) "
                                "received NOTHING for the whole of %s — the "
                                "pinned session id is probably stale",
                                sid_at_send,
                                log_id,
                            )
                        detail = (
                            "no closing marker and no active main turn"
                            if ever_saw_r_marker
                            else "no reply markers ever appeared — the reply "
                            "may be ready in the terminal but its delivery "
                            "markers were lost"
                        )
                        return AskResult("timeout", detail=detail)

                # Stall model: silence is NOT a hang signal — a quiet task
                # still shows the working spinner. Stuck == no visible turn
                # (and no completion) for longer than stall_timeout.
                # Two liveness signals: the recognized spinner, OR a growing
                # transcript — pane.log growth is version-proof and survives
                # any future change to the spinner's on-screen text.
                # is_working_progressing (not the plain is_working) so a
                # background-agent/elapsed-timer frame that stops changing
                # (dead subagent, process wedged in D-state) is NOT treated
                # as ongoing liveness forever — found in review 2026-08-20.
                log_size = self._pane_log_size()
                # Bound how long the static "esc to interrupt" hint alone may
                # stand in for liveness: a pane that has not
                # changed a byte AND whose log has not grown for
                # STATIC_TRUST_WINDOW is frozen, not quiet, and the hint stops
                # being accepted as evidence. Both halves matter — a streamed
                # reply is stripped out of chrome, so log growth is what keeps
                # a genuinely live-but-visually-static turn trusted.
                if _chrome(cap) != _chrome(last_cap) or log_size > last_log_size:
                    last_chrome_change_ts = self._clock()
                trust_static = (
                    self._clock() - last_chrome_change_ts
                ) < STATIC_TRUST_WINDOW
                if (
                    is_working_progressing(cap, last_cap, trust_static=trust_static)
                    or log_size > last_log_size
                ):
                    last_active = self._clock()
                last_log_size = log_size
                last_cap = cap
                if self._clock() - last_active > self._stall_timeout:
                    # Escape (interrupt()), NOT C-c: C-c is deliberately never
                    # used here — on an idle prompt it starts Claude Code's
                    # double-C-c exit sequence, and mid-generation it can
                    # still kill the answer before the closing marker is
                    # written. Escape only ever cancels the response.
                    #
                    # This was dropped entirely at one point (leaving the
                    # turn to finish on its own, recovered later by the
                    # orphan poller), but review found a worse failure mode:
                    # a wedged turn left running is exactly what later trips
                    # watchdog._is_hung(), whose only recovery is
                    # force_recover() — `tmux kill-session`, destroying the
                    # long-lived brain session and its entire context, not
                    # just the one stuck turn. Interrupting here first gives
                    # the pane a chance to land back at idle on its own, so
                    # the far more expensive kill-session path is a last
                    # resort rather than the routine outcome of a stall.
                    #
                    # inflight is cleared here (not left as a stuck marker):
                    # leaving a non-maint inflight behind while the lock is
                    # released would make is_steerable_turn() report True for
                    # a turn we just told the CLI to abandon, so a message
                    # arriving moments later would be steered into a turn
                    # nobody is waiting on any more.
                    self.interrupt()
                    logger.warning(
                        "no liveness signal for %ss on %s — interrupted and "
                        "released the lock",
                        self._stall_timeout,
                        log_id,
                    )
                    self._inflight.unlink(missing_ok=True)
                    return AskResult("error", detail="session stalled (no active turn)")
                self._sleep(self._poll_interval)

            # timed out: prompt is still physically in the pane → keep inflight
            if reply_tail is not None and not reply_tail.saw_records:
                # Same stale-pin alarm as the ceiling branch above, for the
                # one path that skips it: a turn whose MAIN spinner keeps
                # reading as active never reaches the ceiling (it needs
                # no_main_turn_ceiling seconds with no main-turn activity)
                # and exits here instead. A live turn against a transcript
                # that receives nothing is exactly the stale-pin shape, and
                # it must not be the one case that says nothing (blind
                # review, round 2).
                logger.error(
                    "the transcript this session is pinned to (%s) received "
                    "NOTHING for the whole of %s — the pinned session id is "
                    "probably stale",
                    sid_at_send,
                    log_id,
                )
            return AskResult("timeout", detail=f"no reply in {timeout}s")
