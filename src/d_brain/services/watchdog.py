"""Liveness watchdog for the persistent Claude session.

Runs as its own systemd --user service in a SEPARATE slice from the session
(so an OOM kill of the brain doesn't take the watchdog with it). Each tick it
decides one of:

- disk_full        → alert + STOP (a restart can't fix a full disk)
- recovered_dead   → session gone → force_recover + alert
- rate_limited     → subscription limit hit → do NOT kill; wait for the reset
- limit_resumed    → the banner's reset time passed → nudge the session awake
- logged_out       → auth lost → alert (needs re-login); do NOT kill
- recovered_hung   → wedged → force_recover + alert
- recover_deferred → wedged but a live request holds the lock → retry next tick
- healthy          → nothing to do

Hang model: hung == pane state is NOT serviceable (not READY/RATE/LOGGED_OUT)
AND no new bytes have flowed to pane.log for stall_threshold. This catches a
wedged request AND a stuck startup, never kills a long-but-live task (pane.log
keeps growing), and never kills a healthy idle READY session that merely left
an orphan inflight marker (which is cleared on READY). ask() also self-detects
stalls; the watchdog is the second line when the bot process itself died.

Alerts are debounced: a level-triggered fault (disk/logged-out) alerts once
per cooldown, and re-fires after the session returns to a good state.
"""

import logging
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from d_brain.services import long_run
from d_brain.services.systemd_notify import notify, watchdog_interval
from d_brain.services.tmux_parse import (
    PaneState,
    is_main_turn_active,
    reset_epoch,
    unmarked_reset_banner,
)
from d_brain.services.transcript import latest_context_tokens

logger = logging.getLogger(__name__)

# M3 fix (Fable audit fix round, 2026-08-22): the project's own allow-listed
# Telegram HTML tag set (see CLAUDE.md's "Report Format" — b, i, code, pre,
# a). Orphan replies delivered through THIS module's alert_fn go out via
# `_telegram_alerter`'s raw httpx POST, which sets no `parse_mode` at all —
# unlike chat_session.py's normal send, which inherits bot/main.py's global
# `parse_mode=HTML` default. Left unstripped, an orphan reply carrying the
# "⚠️ <i>ответ восстановлен без закрывающего маркера</i>" salvage prefix (or
# any model reply that itself used the report's b/i/code/pre/a markup) shows
# the literal tag characters to the user instead of being rendered. Scoped
# to just THIS tag set — not a generic "strip everything that looks like a
# tag" — because other watchdog alert strings (e.g. delivery_guard's
# escalation message) contain literal `<sha>` placeholder text that must
# reach Telegram unchanged; blanket-enabling parse_mode=HTML on the shared
# alerter instead would make THOSE messages fail Telegram's HTML parser
# (unknown tag `<sha>`) and be silently dropped, which is worse than the bug
# being fixed here.
_REPORT_HTML_TAG_RE = re.compile(r"</?(?:b|i|code|pre|a)(?:\s[^>]*)?>", re.IGNORECASE)


def _strip_report_html(text: str) -> str:
    """Remove the allow-listed report HTML tags from ``text`` — see
    ``_REPORT_HTML_TAG_RE`` above. Never raises; a plain string in, a plain
    string out."""
    return _REPORT_HTML_TAG_RE.sub("", text)

DEFAULT_TICK = 15.0
DEFAULT_STALL_THRESHOLD = 300.0  # 5 min stuck without visible work ⇒ wedged
DEFAULT_MIN_DISK = 500_000_000  # 500 MB
DEFAULT_ALERT_COOLDOWN = 3600.0  # first re-alert of a persistent fault: 1h
DEFAULT_ALERT_COOLDOWN_MAX = 12 * 3600.0  # back-off cap (doubles 1h→2h→…→12h)
# Shortest gap between two wake-up nudges of a rate-limited session. Doubles
# on each consecutive attempt (the parsed reset time can be wrong, or the
# limit can be re-hit immediately) so a stuck limit costs a handful of pokes.
DEFAULT_NUDGE_COOLDOWN = 600.0
DEFAULT_NUDGE_COOLDOWN_MAX = 3600.0
# Fallback when the banner carries no readable reset time: nudge anyway after
# this long rather than sit parked forever, which is the 2026-08-20 incident.
DEFAULT_LIMIT_MAX_WAIT = 6 * 3600.0
# R3 (Fable audit): the incident registry measured a ~10x jump in the
# closing-marker drop rate at this boundary (1.3% below vs ~12.2%/11.6% in
# the 400-600k/600k+ buckets) — notification only, no automatic /clear: that
# stays the user's call per durable-state-first (brain-system.md).
DEFAULT_CONTEXT_ALERT_TOKENS = 400_000
# B2 fix (Fable audit fix round, 2026-08-22): consecutive LOW ticks required
# before the one-time latch is allowed to re-arm (see _check_context_size).
# Even after summing cache_read + cache_creation + input tokens (the B2 fix
# to transcript.latest_context_tokens), a single noisy/atypical low reading
# (a transcript-shape surprise, a race mid-write, a record this module
# doesn't yet know to special-case) must not immediately reopen the latch —
# that was exactly the bug this round fixes (cache-rewrite turns un-arming
# it on the read-alone metric, ~2x/day on real data). 3 ticks at
# DEFAULT_TICK (15s) is ~30-45s of sustained low readings: long enough that
# one anomalous sample can't flip it, short enough that a genuine drop (a
# real /clear, which leaves the context low on EVERY subsequent tick, not
# just one) re-arms within under a minute rather than being noticeably
# delayed.
DEFAULT_CONTEXT_REARM_TICKS = 3
# agent-infra-backlog item 22 (2026-09): how long an UNATTENDED long turn
# (main turn active on the pane, but the ask-lock free — the shape of a
# multi-level autonomous agent cascade) runs before the watchdog sends one
# heads-up notice. See long_run.py's module docstring for the marker
# contract and claude_session.ask()'s busy-active fast path that reads it.
# 0.0 disables long-run tracking entirely (see Watchdog.__init__).
DEFAULT_LONG_RUN_ALERT = 900.0
# Step E (optional, default OFF): when > 0, a one-shot in-pane reminder is
# steered into the session once an unattended run passes this (larger)
# threshold, asking it to close the turn and dispatch remaining work to
# background agents. 0.0 means this code path never fires — see
# Watchdog._track_long_run.
DEFAULT_LONG_RUN_NUDGE = 0.0
# agent-infra-backlog item 29 (2026-09): HARD cap. An unattended turn of the
# main session that runs longer than this is closed automatically (one
# interrupt() per run) and the owner is told why. The rule this encodes is
# the owner's own: a long background job belongs to an agent, the main
# session is loaded only with short work and supervision — so a turn that
# outlives the cap is a failure mode, not work in progress.
#
# THRESHOLD ORDER (all measured from the same "unattended run started"
# instant): alert (DEFAULT_LONG_RUN_ALERT, 900s — one heads-up notice)
# < nudge (DEFAULT_LONG_RUN_NUDGE, off by default — one in-pane reminder to
# self-close) < max (this, 1800s — the watchdog closes it). Configuring them
# out of order is not fatal, just nonsense: the earlier stages would fire
# after the turn was already closed. 0.0 disables the auto-close entirely
# (rollback switch); the alert and the marker are unaffected.
DEFAULT_LONG_RUN_MAX = 1800.0

_SERVICEABLE = {
    PaneState.READY,
    PaneState.RATE_LIMITED,
    PaneState.LOGGED_OUT,
}


class Watchdog:
    def __init__(
        self,
        session: Any,
        *,
        runtime_dir: Path,
        disk_free_fn: Callable[[], int] | None = None,
        clock_fn: Callable[[], float] = time.time,
        alert_fn: Callable[[str], object] = lambda _m: None,
        sleep_fn: Callable[[float], None] = time.sleep,
        tick: float = DEFAULT_TICK,
        stall_threshold: float = DEFAULT_STALL_THRESHOLD,
        min_disk_bytes: int = DEFAULT_MIN_DISK,
        alert_cooldown: float = DEFAULT_ALERT_COOLDOWN,
        alert_cooldown_max: float = DEFAULT_ALERT_COOLDOWN_MAX,
        nudge_cooldown: float = DEFAULT_NUDGE_COOLDOWN,
        nudge_cooldown_max: float = DEFAULT_NUDGE_COOLDOWN_MAX,
        limit_max_wait: float = DEFAULT_LIMIT_MAX_WAIT,
        delivery_guard: Any | None = None,
        context_alert_tokens: int = DEFAULT_CONTEXT_ALERT_TOKENS,
        context_rearm_ticks: int = DEFAULT_CONTEXT_REARM_TICKS,
        long_run_alert_seconds: float = DEFAULT_LONG_RUN_ALERT,
        long_run_nudge_seconds: float = DEFAULT_LONG_RUN_NUDGE,
        long_run_max_seconds: float = DEFAULT_LONG_RUN_MAX,
    ) -> None:
        self.session = session
        self.runtime_dir = Path(runtime_dir)
        self._disk_free_fn = disk_free_fn or (
            lambda: shutil.disk_usage(self.runtime_dir).free
        )
        self._clock = clock_fn
        self._alert_fn = alert_fn
        self._sleep = sleep_fn
        self._tick = tick
        self._stall_threshold = stall_threshold
        self._min_disk = min_disk_bytes
        self._alert_cooldown = alert_cooldown
        self._alert_cooldown_max = alert_cooldown_max
        self._nudge_cooldown = nudge_cooldown
        self._nudge_cooldown_max = nudge_cooldown_max
        self._limit_max_wait = limit_max_wait
        self._inflight = self.runtime_dir / "inflight"
        self._status = self.runtime_dir / "STATUS.md"
        # Persists _limited_since across a watchdog restart (deploy, crash,
        # `systemctl restart`) while the session is still parked at the same
        # banner. Without this, every restart re-anchored the "first seen"
        # moment to "now" — and if the banner's reset time had ALREADY passed
        # by then (the exact 2026-08-20 incident: diagnosed after the
        # midnight reset had come and gone), _reset_deadline() rolled the
        # candidate to the SAME time tomorrow, so a restart mid-outage could
        # add up to 24h to the wait rather than shortening it.
        self._limited_since_file = self.runtime_dir / "limited_since"
        # Watches the OUTCOME of the bot's turns, not the brain's liveness —
        # the 2026-08-20 blind spot. Optional so every existing test that
        # builds a Watchdog keeps building one.
        self._delivery_guard = delivery_guard
        self._context_alert_tokens = context_alert_tokens
        self._context_rearm_ticks = context_rearm_ticks
        # R3: persisted, genuinely ONE-TIME latch — not the exponential
        # backoff _maybe_alert gives level-triggered faults below. Rearms
        # itself once the context measurably drops back under the threshold
        # (e.g. after a /clear), so a LATER crossing alerts again.
        self._context_alert_sent = self.runtime_dir / "context_alert_sent"
        # B2 fix: consecutive-low-tick counter gating the rearm above (see
        # DEFAULT_CONTEXT_REARM_TICKS). In-memory only — a watchdog restart
        # mid-streak merely costs a few extra ticks before rearming is
        # judged safe again, never a correctness issue (same tradeoff as
        # ClaudeSession._orphan_open_seen).
        self._low_context_streak = 0
        self._last_alert_key: str | None = None
        self._last_alert_ts = 0.0
        self._alert_repeats = 0
        self._stuck_since: float | None = None
        # Rate-limit bookkeeping. _limited_since is when THIS watchdog first
        # saw the banner: the banner names a bare wall-clock time ("resets
        # 12am"), so the only way to turn it into an instant is "the first
        # such time at or after we saw it".
        self._limited_since: float | None = None
        self._last_nudge_ts = 0.0
        self._nudge_attempts = 0
        # agent-infra-backlog item 22: unattended long-run tracking (see
        # long_run.py). 0.0 disables the whole feature — no marker writes,
        # no notices.
        self._long_run_alert_seconds = long_run_alert_seconds
        # Step E (optional, default OFF): one-shot in-pane nudge threshold.
        # 0.0 (the default) means this never fires.
        self._long_run_nudge_seconds = long_run_nudge_seconds
        self._long_run_nudge_sent = False
        # Item 29: hard cap. 0.0 (or a non-positive value) never closes
        # anything. Both pieces of state are per-RUN latches reset on
        # 'started'/'ended', exactly like _long_run_nudge_sent — so at most
        # ONE interrupt() is ever issued for a given unattended run, however
        # many ticks observe it past the cap.
        self._long_run_max_seconds = long_run_max_seconds
        self._long_run_closed = False
        # When the auto-close (or the last escalation reminder about a run
        # that refused to close) was announced. Rate-limits the escalation to
        # at most one message per _long_run_max_seconds — a turn that ignores
        # the interrupt must not turn into a notification flood, and must
        # NEVER escalate into force_recover/kill: that would destroy the
        # session and the very context whose partial work we are preserving.
        self._long_run_closed_ts = 0.0

    def _is_hung(self, state: PaneState) -> bool:
        # Hang model (paired with ask()'s stall detector): silence is NOT a
        # signal — a long quiet task still shows the working spinner. Hung ==
        # non-serviceable AND no visible work, PERSISTING past the threshold.
        if state in _SERVICEABLE or self.session.is_working():
            self._stuck_since = None
            return False
        now = self._clock()
        if self._stuck_since is None:
            self._stuck_since = now
            return False
        return now - self._stuck_since >= self._stall_threshold

    def _maybe_alert(self, key: str, msg: str) -> None:
        now = self._clock()
        if self._last_alert_key == key:
            # Same persistent fault: back off exponentially (1h, 2h, 4h … up to
            # the cap) so an overnight outage sends a few escalating reminders
            # instead of one identical message every hour.
            cooldown = min(
                self._alert_cooldown * (2**self._alert_repeats),
                self._alert_cooldown_max,
            )
            if now - self._last_alert_ts < cooldown:
                return
            self._alert_repeats += 1
        else:
            self._alert_repeats = 0
        self._alert_fn(msg)
        self._last_alert_key = key
        self._last_alert_ts = now

    def _note_good(self) -> None:
        # Returning to a good state re-arms alerts for the next incident.
        self._last_alert_key = None
        self._alert_repeats = 0

    def _write_status(self, state: str) -> None:
        try:
            self._status.write_text(f"state: {state}\nchecked_at: {self._clock()}\n")
        except OSError as exc:
            logger.warning("could not write STATUS.md: %s", exc)

    # ── subscription limit ───────────────────────────────────────────

    def _reset_deadline(self, now: float) -> float | None:
        """Epoch of the limit's reset, or None if the banner doesn't say.

        The banner gives a time of day with no date, so it is anchored to the
        moment we first saw it: the reset is the first occurrence of that UTC
        time at or after that moment.
        """
        try:
            cap = self.session.capture_text()
            deadline = reset_epoch(cap, seen_at=self._limited_since or now)
        except Exception:  # noqa: BLE001 — never let a tick die on parsing
            logger.warning("could not read pane for reset time")
            return None
        if deadline is None:
            # Ambiguous by itself (no banner / no time clause / an unmarked
            # time) — this diagnostic narrows it to the one case worth a
            # loud log: a reset time IS on screen but without the UTC marker
            # parse_reset_time() requires, so the caller is about to fall
            # back to limit_max_wait with no explanation at all (review
            # 2026-08-20).
            unmarked = unmarked_reset_banner(cap)
            if unmarked is not None:
                logger.warning(
                    "rate-limit banner has a reset time with no UTC marker — "
                    "treating as unparseable, falling back to limit_max_wait: %r",
                    unmarked,
                )
            return None
        return deadline

    def _load_limited_since(self) -> float | None:
        try:
            return float(self._limited_since_file.read_text().strip())
        except (OSError, ValueError):
            return None

    def _persist_limited_since(self) -> None:
        try:
            self._limited_since_file.write_text(f"{self._limited_since}\n")
        except OSError as exc:
            logger.warning("could not persist limited_since: %s", exc)

    def _clear_limit_state(self) -> None:
        self._limited_since = None
        self._nudge_attempts = 0
        self._limited_since_file.unlink(missing_ok=True)

    def _handle_rate_limited(self) -> str:
        """Wait out a subscription limit, then wake the session up.

        Claude Code parks at the banner and never re-checks the clock — it
        only acts on the next input. Before this, a limit that reset at
        midnight left the brain silent until the user happened to write (verified
        2026-08-20: a manual send-keys revived it instantly).
        """
        now = self._clock()
        first_sighting = self._limited_since is None
        if first_sighting:
            self._limited_since = self._load_limited_since()
            if self._limited_since is None:
                self._limited_since = now
            self._persist_limited_since()
        reset_at = self._reset_deadline(now)
        # due is capped to limit_max_wait UNCONDITIONALLY, not chosen as
        # "reset_at if parsed else the fallback": a parsed reset_at can still
        # be far in the future relative to when we first saw the banner — the
        # banner names a bare wall-clock time with no date, so if that time
        # had ALREADY passed at first sighting, _reset_deadline() rolls it to
        # the SAME time tomorrow (correctly, for a banner seen just before
        # the reset). But a watchdog that only starts diagnosing well AFTER
        # the reset has passed — an idle brain sitting at the banner across a
        # deploy or restart, or diagnosed hours late — would otherwise wait
        # up to ~24h instead of the intended handful of hours.
        due = self._limited_since + self._limit_max_wait
        if reset_at is not None:
            due = min(due, reset_at)
        if first_sighting:
            logger.info(
                "rate limit banner first seen at %.0f (reset_at=%s, due=%.0f)",
                self._limited_since,
                f"{reset_at:.0f} UTC" if reset_at is not None else "unparsed",
                due,
            )
        if now < due:
            self._write_status(f"rate_limited (retry at {due:.0f})")
            return "rate_limited"

        cooldown = min(
            self._nudge_cooldown * (2**self._nudge_attempts), self._nudge_cooldown_max
        )
        if self._last_nudge_ts and now - self._last_nudge_ts < cooldown:
            self._write_status("rate_limited (nudge cooling down)")
            return "rate_limited"
        # A live ask() holding the pane lock means the session is already
        # working — nothing to wake. Try again next tick.
        if not self.session.nudge("Continue"):
            self._write_status("rate_limited (busy, nudge deferred)")
            return "rate_limited"
        self._last_nudge_ts = now
        self._nudge_attempts += 1
        logger.info(
            "nudged rate-limited session awake (attempt %d, due was %.0f)",
            self._nudge_attempts,
            due,
        )
        self._maybe_alert(
            "limit_resumed",
            "▶️ Лимит подписки обновился — разбудил сессию, продолжаю работу.",
        )
        self._write_status("limit_resumed")
        return "limit_resumed"

    # ── R3: context-size notification (no automatic action) ────────────

    def _check_context_size(self) -> None:
        """One-time alert when the session's context crosses
        ``context_alert_tokens`` — see DEFAULT_CONTEXT_ALERT_TOKENS. Runs
        every tick, independent of this tick's verdict (mirrors the
        delivery_guard.tick() call above it in check_once): never raises,
        never blocks a real fault from being reported.
        """
        try:
            transcript = self.session.current_transcript_path()
        except Exception:  # noqa: BLE001 — a duck-typed session may not
            # implement this (older tests, alternate callers) — treat as
            # "nothing to check", never crash the tick over it.
            return
        if transcript is None or not Path(transcript).exists():
            return
        try:
            tokens = latest_context_tokens(transcript)
        except Exception:  # noqa: BLE001 — diagnostic only, never fatal
            logger.warning("R3 context check failed", exc_info=True)
            return
        if tokens is None:
            return
        if tokens < self._context_alert_tokens:
            # B2 fix: require several CONSECUTIVE low ticks before rearming
            # — a single low reading (transcript-shape surprise, a record
            # this module doesn't special-case, a race mid-write) must not
            # immediately reopen the latch. See DEFAULT_CONTEXT_REARM_TICKS.
            self._low_context_streak += 1
            if self._low_context_streak >= self._context_rearm_ticks:
                # Sustained drop back below the threshold (e.g. a /clear
                # happened) — rearm so a LATER crossing alerts again.
                self._context_alert_sent.unlink(missing_ok=True)
            return
        self._low_context_streak = 0
        if self._context_alert_sent.exists():
            return  # already alerted for this crossing
        try:
            self._context_alert_sent.write_text(f"{tokens}\n")
        except OSError:
            logger.warning("could not persist context_alert_sent latch")
        self._alert_fn(
            f"📊 Контекст сессии перевалил за ~{tokens // 1000}k токенов. "
            "По данным реестра инцидентов пропуск закрывающего маркера "
            "ответа подскакивает с ~1.3% (ниже 400k) до ~12% (выше) — "
            "стоит подумать про /clear, но решение за тобой, "
            "автоматически не чищу."
        )
        logger.info("R3: context alert sent at %d tokens", tokens)

    # ── unattended long-run tracking (agent-infra-backlog item 22) ──────

    def _track_long_run(self) -> None:
        """Track an unattended long-running turn — see long_run.py's module
        docstring for the full picture. Runs every tick, independent of this
        tick's verdict (same contract as ``_check_context_size`` above):
        never raises, never changes what ``check_once`` returns. Purely for
        user visibility (fast honest notices) and to let ``ask()`` short-
        circuit its own busy-wait via the marker this writes — it never
        itself decides anything is a fault.

        Three thresholds hang off the same tracked run, in this order (see
        ``DEFAULT_LONG_RUN_MAX``): ``alert`` → one heads-up message;
        ``nudge`` (off by default) → one in-pane reminder to self-close;
        ``max`` → ``_enforce_long_run_cap`` closes the turn. Item 29's cap
        is the only one of the three that CHANGES the session's state
        rather than just reporting on it.

        Defensive/duck-typed like the rest of this method's siblings: a fake
        or minimal session object used in tests may not implement every real
        method, and that must never crash a tick.
        """
        if self._long_run_alert_seconds <= 0:
            return
        try:
            cap = self.session.capture_text()
            main_turn_active = is_main_turn_active(cap)
            attended = self.session.is_turn_active()
        except Exception:  # noqa: BLE001 — duck-typed session, never fatal
            return
        now = self._clock()
        previous = long_run.read(self.runtime_dir)
        new_state, event = long_run.next_state(
            previous,
            main_turn_active=main_turn_active,
            attended=attended,
            now=now,
            alert_after=self._long_run_alert_seconds,
        )
        long_run.write(self.runtime_dir, new_state)

        if event in ("started", "ended"):
            self._long_run_nudge_sent = False
            self._long_run_closed = False
            self._long_run_closed_ts = 0.0

        if event == "started":
            logger.info("long-run: unattended long turn started")
        elif event == "alert":
            elapsed = now - previous.since
            minutes = max(1, round(elapsed / 60))
            logger.info("long-run: alert at ~%.0fs elapsed", elapsed)
            self._alert_fn(
                f"🛠 Идёт длинная автономная задача — сессия занята уже "
                f"~{minutes} мин. Обычные сообщения в это время могут "
                "получать «занято». Канал доставки при этом исправен."
            )
        elif event == "ended":
            elapsed = now - previous.since
            minutes = max(1, round(elapsed / 60))
            logger.info("long-run: ended after ~%.0fs elapsed", elapsed)
            # F2 fix (blind-review round, 2026-09): only announce the end if
            # we actually announced the start. `previous.alerted` is False
            # for any run that never crossed `alert_after` — a brief
            # unattended blip (a proactive self-initiated turn, a leftover
            # turn clearing right after a stall) would otherwise send an
            # unsolicited "your long task finished" message for something
            # the user was never told had started, however short. Only the
            # "started"→"alert" pair is user-visible; "ended" mirrors that.
            if previous.alerted:
                self._alert_fn(
                    f"✅ Длинная автономная задача завершилась (~{minutes} мин) "
                    "— сессия снова свободна."
                )

        # Step E (optional, default OFF via DEFAULT_LONG_RUN_NUDGE = 0.0):
        # one-shot in-pane reminder once an unattended run passes the
        # (larger) nudge threshold — never fires with the flag at 0.
        if (
            self._long_run_nudge_seconds > 0
            and new_state.since > 0
            and not self._long_run_nudge_sent
            and now - new_state.since >= self._long_run_nudge_seconds
        ):
            try:
                self.session.steer(
                    "🛠 Watchdog notice: this turn has been open for a "
                    "while during unattended background work. Per "
                    "deploy/brain-system.md's cascade rule — close this "
                    "turn now (emit the closing reply marker if one is "
                    "open) and dispatch any remaining work to background "
                    "agents instead of continuing to hold it open."
                )
                self._long_run_nudge_sent = True
                logger.info("long-run: sent one-shot nudge into the pane")
            except Exception:  # noqa: BLE001 — best effort, never fatal
                logger.warning("long-run: nudge steer failed", exc_info=True)

        self._enforce_long_run_cap(new_state.since, now)

    def _enforce_long_run_cap(self, since: float, now: float) -> None:
        """Hard cap on an unattended run (agent-infra-backlog item 29).

        Last stage of the threshold ladder documented on
        ``DEFAULT_LONG_RUN_MAX``: alert (heads-up) < nudge (ask the session
        to self-close, off by default) < max (THIS — close it ourselves).

        Only ever reached from ``_track_long_run``, which means it only ever
        sees an UNATTENDED run: ``long_run.next_state`` refuses to track a
        turn whose ask-lock is held, so an ordinary chat turn the user is
        sitting in front of — however long it runs — can never be closed by
        this path. That is deliberate, not an oversight: the user waiting on
        their own answer is exactly who this cap exists to protect.

        Two-stage, and the second stage is deliberately toothless:

        1. First tick past the cap: ONE ``interrupt()`` (a TUI Escape / a
           SIGINT to the exec process — the same thing ``/stop`` does) plus
           one message explaining what happened and where the partial work
           is.
        2. If later ticks still see the SAME run (the turn ignored the
           interrupt): at most one reminder per ``_long_run_max_seconds``,
           and nothing else. No repeated interrupts, and explicitly no
           ``force_recover``/kill — recovering the session would destroy the
           conversation context that holds the very work this message is
           telling the owner about.
        """
        if self._long_run_max_seconds <= 0 or since <= 0:
            return
        elapsed = now - since
        if elapsed < self._long_run_max_seconds:
            return
        minutes = max(1, round(elapsed / 60))

        if self._long_run_closed:
            # Stage 2: it did not close. Remind, rarely; never re-interrupt.
            if now - self._long_run_closed_ts < self._long_run_max_seconds:
                return
            self._long_run_closed_ts = now
            logger.warning(
                "long-run: turn still running %.0fs after auto-close", elapsed
            )
            self._alert_fn(
                f"⚠️ Ход основной сессии не закрылся после автоматического "
                f"прерывания — идёт уже ~{minutes} мин. Сессию принудительно "
                "не перезапускаю: это уничтожило бы её контекст вместе с "
                "недоделанной работой. Прервать вручную — /stop."
            )
            return

        # Stage 1: one interrupt, one honest message.
        self._long_run_closed = True
        self._long_run_closed_ts = now
        # Duck-typed session (tests, minimal fakes) like every sibling in
        # this method's neighborhood: a tick must never die here.
        try:
            self.session.interrupt()
            closed = True
        except Exception:  # noqa: BLE001 — best effort, never fatal
            closed = False
            logger.warning("long-run: interrupt failed", exc_info=True)
        logger.info(
            "long-run: hard cap reached at ~%.0fs elapsed (interrupt %s)",
            elapsed,
            "sent" if closed else "FAILED",
        )
        if closed:
            self._alert_fn(
                f"⏹ Ход основной сессии шёл ~{minutes} мин — закрыл его "
                "автоматически. Длинная работа должна уходить фоновым "
                "агентам, а не занимать основную сессию. Что успело "
                "записаться — лежит в файлах вольта. Если результат всё ещё "
                "нужен — напиши ещё раз, вынесу это в агента."
            )
        else:
            self._alert_fn(
                f"⚠️ Ход основной сессии идёт ~{minutes} мин — это дольше "
                "лимита, но закрыть его автоматически не получилось. "
                "Длинная работа должна уходить фоновым агентам. Прервать "
                "вручную — /stop."
            )

    def _deliver_notices(self) -> None:
        """Forward the session's owner notices (e.g. "conversation context
        reset: the pane was parked", agent-infra-backlog item 28). Parking
        can happen at bot start-up or in force_recover(), with no chat reply
        to carry the notice — this tick is what makes it reach the owner.
        Best-effort; a session without notices (Codex driver) is skipped.

        A notice whose send fails (alert_fn raises, or returns False — the
        Telegram alerter does that when admin_chat_id is empty or the API
        call fails) is handed back to the session together with every notice
        after it, so a later tick — or the next chat reply — can deliver it.
        pop_notices() empties the queue before the send is known to work;
        without the requeue one failed POST lost the notice for good (review
        2026-09-19)."""
        pop = getattr(self.session, "pop_notices", None)
        if pop is None:
            return
        try:
            notices = list(pop())
        except Exception:  # noqa: BLE001 — never break the liveness tick
            logger.warning("could not read session notices", exc_info=True)
            return
        for i, text in enumerate(notices):
            try:
                sent = self._alert_fn(text) is not False
            except Exception:  # noqa: BLE001
                logger.warning("owner notice send failed", exc_info=True)
                sent = False
            if not sent:
                self._requeue_notices(notices[i:])
                return

    def _requeue_notices(self, texts: list[str]) -> None:
        requeue = getattr(self.session, "requeue_notices", None)
        try:
            if requeue is None:
                raise AttributeError("session cannot take notices back")
            requeue(texts)
            logger.warning(
                "owner notice not delivered — %d kept for a later try", len(texts)
            )
        except Exception:  # noqa: BLE001 — never break the liveness tick
            for text in texts:
                logger.error("owner notice lost: %s", text, exc_info=True)

    def _recover(self, reason: str, alert_msg: str) -> str:
        if self.session.force_recover():
            self._maybe_alert(f"recovered_{reason}", alert_msg)
            self._write_status(f"recovered_{reason}")
            return f"recovered_{reason}"
        # A live request holds the lock — don't claim a restart happened.
        self._write_status("recover_deferred")
        return "recover_deferred"

    def check_once(self) -> str:
        """One liveness tick. Returns the decision string."""
        if self._disk_free_fn() < self._min_disk:
            self._maybe_alert(
                "disk_full", "🔴 Диск переполнен — dbrain не работает (dbrain repair)."
            )
            self._write_status("disk_full")
            return "disk_full"

        # Runs on every tick and never changes this tick's verdict: the brain
        # can be perfectly READY while the bot in front of it delivers
        # nothing, which is exactly what happened on 2026-08-20.
        if self._delivery_guard is not None:
            self._delivery_guard.tick()
        self._deliver_notices()
        self._check_context_size()
        self._track_long_run()

        if not self.session.is_healthy():
            return self._recover("dead", "♻️ Мозг был мёртв — перезапустил.")

        state = self.session.current_state()
        if state == PaneState.RATE_LIMITED:
            return self._handle_rate_limited()
        self._clear_limit_state()
        if state == PaneState.LOGGED_OUT:
            self._maybe_alert(
                "logged_out",
                "🔑 Claude разлогинился — нужен повторный вход (dbrain login).",
            )
            self._write_status("logged_out")
            return "logged_out"

        if self._is_hung(state):
            return self._recover("hung", "♻️ Мозг завис — перезапустил.")

        if state == PaneState.READY:
            try:
                turn_active = self.session.is_turn_active()
            except Exception:  # duck-typed session in tests may not implement this
                turn_active = False
            if not turn_active:
                self._inflight.unlink(missing_ok=True)  # clear any orphan marker
            # Idle + no live ask() in flight is exactly when a background
            # subagent's self-marked reply (see pop_orphan_replies) would
            # otherwise sit undelivered until the human happened to write
            # in again. Best-effort: a failed send is dropped, not retried
            # (retrying risks duplicate delivery far more than it's worth).
            # Plural: several replies can be waiting (a subagent finished
            # while a later turn ran), and delivering only the newest is how
            # earlier ones went missing entirely.
            for orphan in self.session.pop_orphan_replies():
                try:
                    # M3 fix: this alert path has no HTML rendering (see
                    # _strip_report_html's docstring) — strip the tags
                    # instead of leaking them to the user literally.
                    self._alert_fn(_strip_report_html(orphan))
                    # The one line that tells 2026-08-20's postmortem apart
                    # from "the mechanism is silently doing nothing": without
                    # it, a healthy tick and a tick that just delivered a
                    # recovered reply were indistinguishable in the logs.
                    logger.info(
                        "watchdog delivered orphan reply (%d chars)", len(orphan)
                    )
                except Exception:
                    # ERROR, not warning (review 2026-08-20): the rid was
                    # already marked handled BEFORE this send by
                    # pop_orphan_replies(), so a failure here loses the
                    # reply permanently — it will never be retried or
                    # resurfaced. Deliberately no rid here (pop_orphan_replies
                    # only hands back the body, see its docstring for why);
                    # correlate by timestamp with the "delivered orphan
                    # reply rid=..." INFO line pop_orphan_replies() just
                    # logged for this same body.
                    logger.error(
                        "orphan reply delivery failed, %d chars lost",
                        len(orphan),
                        exc_info=True,
                    )
        self._note_good()
        self._write_status("healthy")
        return "healthy"

    def run(self) -> None:  # pragma: no cover - long-running loop
        """Main loop: tick, ping systemd watchdog, sleep."""
        notify("READY=1")
        interval = min(self._tick, watchdog_interval(self._tick))
        while True:
            try:
                self.check_once()
            except Exception:
                logger.exception("watchdog tick failed")
            notify("WATCHDOG=1")
            self._sleep(interval)


def _telegram_alerter(settings) -> Callable[[str], bool]:  # pragma: no cover
    import httpx

    def send(msg: str) -> bool:
        """False when nothing reached Telegram (callers that must not lose a
        message, like Watchdog._deliver_notices, keep it; the rest ignore
        the result)."""
        if not settings.admin_chat_id:
            logger.warning("watchdog alert not sent: admin_chat_id is not set")
            return False
        try:
            resp = httpx.post(
                f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                data={"chat_id": settings.admin_chat_id, "text": msg},
                timeout=10,
            )
        except Exception:
            logger.warning("watchdog alert send failed")
            return False
        if not resp.is_success:
            logger.warning("watchdog alert rejected: HTTP %s", resp.status_code)
            return False
        return True

    return send


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    from d_brain import logsafe

    logsafe.install()
    from d_brain.config import get_settings
    from d_brain.services.delivery_guard import DeliveryGuard, systemctl_restarter
    from d_brain.services.runtime import get_session

    settings = get_settings()
    session = get_session(settings)
    alerter = _telegram_alerter(settings)
    Watchdog(
        session,
        runtime_dir=settings.runtime_dir,
        alert_fn=alerter,
        delivery_guard=DeliveryGuard(
            settings.runtime_dir,
            # C3: assembly-level wiring ONLY — no detection/delivery logic in
            # this file is touched. Defaults reproduce today's behavior
            # exactly (project_root → vault_path.parent, bot_unit →
            # dbrain-bot.service, scope → user).
            restart_fn=systemctl_restarter(scope=settings.systemd_scope),
            alert_fn=alerter,
            unit=settings.bot_unit,
            repo_dir=settings.project_root,
        ),
        context_alert_tokens=settings.context_alert_tokens,
        long_run_alert_seconds=settings.long_run_alert_seconds,
        long_run_nudge_seconds=settings.long_run_nudge_seconds,
        long_run_max_seconds=settings.long_run_max_seconds,
    ).run()


if __name__ == "__main__":
    main()
