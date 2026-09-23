"""Drive the brain as a sequence of `codex exec` processes (Codex engine).

Phase 2 of the engine implementation plan,
built on the empirical findings of the phase-0 spike
(~115 real
`codex exec` turns on this exact machine and CLI version, 0.153.4).

This is the second implementation of ``engine.EngineDriver``. It is NOT a
port of ``claude_session.py`` and shares no code with it beyond the outcome
vocabulary (``AskResult``) and three timing constants: the two engines have
genuinely different shapes, and pretending otherwise is how you get a driver
that lies about what it can do.

Shape, and why it is so much simpler than the tmux one
------------------------------------------------------
``ClaudeSession`` keeps ONE long-lived interactive TUI alive in a tmux pane
and reads the answer back off the screen. Everything expensive in that file
— the ``<<<R:id>>>``/``<<<E:id>>>`` marker contract, salvage of unterminated
spans, the no-main-turn ceiling, ``STATIC_TRUST_WINDOW``, orphan bookkeeping
— exists because "the turn is over" has to be *inferred from pixels*.

Codex gives that away for free. ``codex exec --json`` streams JSONL events
(``thread.started``, ``turn.started``, ``item.started``, ``item.completed``,
``turn.completed``) and the spike measured **83 successful turns → 83
``turn.completed`` events, zero misses**. So here:

* end of turn        = the ``turn.completed`` event (not a screen heuristic)
* the reply          = the last ``agent_message`` item's ``text``
* a turn in flight   = the exec process is alive
* interrupt          = SIGINT to that process (spike step 5: process dies on
  its own, rc=1, no ``turn.completed``, and the thread survives — the very
  next ``resume`` works and the partial work is in the transcript)
* multi-turn memory  = ``codex exec resume <thread_id>`` (spike step 2: 55/55
  sequential turns, thread never lost, ~1.19k context tokens per turn)

The marker protocol, salvage, the ceiling and the static-trust window are
therefore ABSENT BY CONSTRUCTION, not "not implemented yet".

Everything below maps a Codex outcome onto the SAME ``AskResult`` statuses
``ClaudeSession`` produces (``ok`` / ``error`` / ``timeout`` / ``busy`` /
``busy_active`` / ``rate_limited`` / ``logged_out``), which is the whole
point of the seam: ``ask_health``, ``delivery_guard`` and the cron statistics
keep working unmodified whichever engine is behind them.

Facts about the CLI this driver depends on (verified, not assumed)
------------------------------------------------------------------
* ``codex exec resume`` does NOT accept ``-s/--sandbox`` or ``-C/--cd``
  ("error: unexpected argument") — the sandbox and working root are fixed at
  thread-creation time. Hence the ``thread_id`` MUST be persisted and the
  first call of a thread is the only one that carries those flags. (Spike
  step 2's "trap that cost me the first run"; re-verified here against
  ``codex exec resume --help`` on 0.153.4.)
* Rate-limit telemetry is NOT in the ``--json`` stream. ``used_percent`` /
  ``resets_at`` / ``rate_limit_reached_type`` live only in the thread's
  rollout file under ``$CODEX_HOME/sessions/YYYY/MM/DD/rollout-*-<id>.jsonl``
  as a ``token_count`` event. ``_rollout_rate_limited()`` below reads it, per
  the spike's explicit warning to phase 2.
* ``--sandbox workspace-write`` restricts WRITES only; reads are unrestricted
  (spike step 3's isolation finding). This driver therefore does not treat
  the sandbox flag as a read boundary and never claims to. The owner removed the
  privacy half of that finding on 2026-09-05 ("пусть читает настоящие данные")
  but explicitly kept the write-protection half, which is why the default
  sandbox here stays ``workspace-write`` rather than ``danger-full-access``.
* stdin must be ``DEVNULL``. With an inherited stdin the CLI prints "Reading
  additional input from stdin..." and waits, which would hang every turn.

Deliberate non-parity, stated plainly rather than faked
-------------------------------------------------------
``steer()`` — NO PARITY. A ``codex exec`` process consumes its prompt at
start-up; there is no mid-turn input channel, and stdin is DEVNULL by
necessity (above). ``is_steerable_turn()`` therefore returns False
unconditionally, which routes ``bot/handlers/chat.py`` into its existing
"background work in progress, resend in a few minutes" branch — the user is
told to resend rather than having their text silently dropped. ``steer()``
itself only ever logs the text at ERROR (so it is recoverable from the
journal) and is unreachable through the bot.

``send_control()`` — different semantics by necessity, see that method.

The tmux watchdog — ``watchdog.py`` classifies ``capture_text()`` with
``tmux_parse`` and calls ``current_state()``/``nudge()``, which are not part
of ``EngineDriver`` precisely because they are tmux concepts. Compatibility
shims exist at the bottom of this class so a Codex session degrades to
"watchdog finds nothing to do" instead of crash-looping on AttributeError,
but a real watchdog for this engine is explicitly out of the compressed
phase-2 scope (agent-infra-backlog item 26).
"""

import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

from d_brain.services.claude_session import (
    DEFAULT_BUSY_WAIT_BUDGET,
    DEFAULT_STALL_TIMEOUT,
    DEFAULT_TIMEOUT,
    AskResult,
)

logger = logging.getLogger(__name__)

__all__ = ["CodexExecDriver", "DEFAULT_INTERRUPT_GRACE", "DEFAULT_SANDBOX"]

# How long a SIGINT'ed turn is given to exit on its own before SIGKILL.
# Spike step 5 measured the process dying by itself on SIGINT (rc=1, no
# turn.completed), so this is a backstop for the pathological case, not the
# expected path.
DEFAULT_INTERRUPT_GRACE = 10.0
# Write protection, NOT read isolation — see the module docstring. Kept as
# the default because the spike proved it actually blocks writes outside the
# workspace (the attempted write to the live bot's cron state was refused).
DEFAULT_SANDBOX = "workspace-write"
# How long the driver waits on a single readline before re-checking its
# deadlines. Small enough that a hard timeout is honored promptly, large
# enough not to spin.
DEFAULT_POLL_INTERVAL = 0.5
# Bytes of the structured turn journal returned by capture_text().
_CAPTURE_TAIL_BYTES = 16384
# Cap on the raw event log; it is forensic material, not state.
_EVENT_LOG_CAP_BYTES = 8 * 1024 * 1024
# Post-SIGINT drain window: how long one read waits, and how many empty reads
# in a row end the drain. Real wall time, not the injected clock — this races
# a dying OS process, not a modelled deadline. ~1s worst case.
_DRAIN_READ_TIMEOUT = 0.05
_DRAIN_MAX_IDLE = 20

# Vendor error text → our status. Order matters: the logged-out signatures
# are checked first because they are the more specific of the two, and a
# genuine auth failure never mentions rate limits.
#
# Sources: the spike's step-1 characterization of the exec error path
# (`turn.failed` + rc=1, taxonomy "rate limit exceeded" / "usage not
# included" / "context window exceeded" / "server overloaded", user-facing
# text "You've hit your usage limit."), plus Codex's own internal detector
# shape (`"429" in msg or "rate limit" in msg or "too many requests" in msg`).
_LOGGED_OUT_MARKERS = (
    "not logged in",
    "no codex credentials",
    "codex login",
    "unauthorized",
    "401",
    "authentication failed",
    "authentication required",
    "token has expired",
    "refresh token",
)
_RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "too many requests",
    "usage limit",
    "usage not included",
    "quota",
)


@dataclass
class _TurnOutcome:
    """What the event stream said about one exec process."""

    completed: bool = False
    reply: str | None = None
    thread_id: str | None = None
    error: str | None = None
    events: int = 0


class _EventReader:
    """Pump a process's stdout into a queue on a daemon thread.

    A thread rather than select/asyncio because ``ask()`` is a synchronous
    call made from ``asyncio.to_thread`` by every caller in this codebase —
    matching ``ClaudeSession.ask``'s contract, which is also blocking.
    """

    def __init__(self, stream, on_line: Callable[[str], None]) -> None:
        self._queue: Queue = Queue()
        self._stream = stream
        self._on_line = on_line
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        try:
            while True:
                line = self._stream.readline()
                if not line:
                    break
                try:
                    self._on_line(line)
                except Exception:  # noqa: BLE001 — journaling must not kill the pump
                    logger.warning("codex: could not journal event line", exc_info=True)
                self._queue.put(line)
        except Exception:  # noqa: BLE001 — a closed pipe is a normal end
            logger.debug("codex: event reader ended", exc_info=True)
        finally:
            self._queue.put(None)  # EOF sentinel

    def next_line(self, timeout: float) -> str | None | object:
        """A line, ``None`` for EOF, or ``_EventReader.NOTHING`` on timeout."""
        try:
            item = self._queue.get(timeout=timeout)
        except Empty:
            return self.NOTHING
        return item

    NOTHING = object()


class CodexExecDriver:
    """One brain "session" backed by per-turn ``codex exec`` processes.

    Constructor deliberately mirrors ``ClaudeSession``'s shape where the
    concept survives (``session_name`` / ``work_dir`` / ``runtime_dir`` /
    ``model`` / injected clock+sleep+rid for tests) and drops what does not
    (``mcp_config``, ``tmux_config``,
    ``system_prompt_file`` → ``instructions_file``).
    """

    def __init__(
        self,
        session_name: str,
        work_dir: Path,
        runtime_dir: Path,
        *,
        instructions_file: Path | None = None,
        model: str | None = None,
        sandbox: str = DEFAULT_SANDBOX,
        codex_bin: str = "codex",
        codex_home: Path | None = None,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
        rid_factory: Callable[[], str] | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        stall_timeout: float = DEFAULT_STALL_TIMEOUT,
        busy_wait_budget: float = DEFAULT_BUSY_WAIT_BUDGET,
        interrupt_grace: float = DEFAULT_INTERRUPT_GRACE,
    ) -> None:
        self.session_name = session_name
        self.work_dir = Path(work_dir)
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        # Same reasoning as ClaudeSession: the dir holds the full transcript
        # and turn state, so it is owner-only, re-asserted every start; a
        # foreign-owned dir degrades to a warning rather than a crash-loop.
        try:
            os.chmod(self.runtime_dir, 0o700)
        except OSError as exc:
            logger.warning(
                "could not restrict %s to owner-only: %s", self.runtime_dir, exc
            )
        self.instructions_file = Path(instructions_file) if instructions_file else None
        self.model = model
        self.sandbox = sandbox
        self.codex_bin = codex_bin
        self.codex_home = Path(codex_home) if codex_home else None
        self._popen = popen
        self._sleep = sleep_fn
        self._clock = clock_fn
        self._rid_factory = rid_factory or (lambda: uuid.uuid4().hex[:8])
        self._poll_interval = poll_interval
        self._stall_timeout = stall_timeout
        self._busy_wait_budget = busy_wait_budget
        self._interrupt_grace = interrupt_grace

        # ── on-disk state (the cross-process contract) ───────────────
        # One turn at a time, across processes — same fcntl-on-a-file
        # mechanism as ClaudeSession's pane.lock, different name because
        # there is no pane. Held for the whole duration of a turn.
        self._turn_lock = self.runtime_dir / "turn.lock"
        # Guards read-modify-write of the orphan queue; deliberately separate
        # from turn.lock for the same reason ClaudeSession keeps state.lock
        # separate — bookkeeping must never queue behind an hour-long turn.
        self._state_lock = self.runtime_dir / "state.lock"
        # request_id + claim time of the turn in flight; drives
        # is_steerable_turn()'s maint check exactly as in ClaudeSession.
        self._inflight = self.runtime_dir / "inflight"
        # THE piece of state that makes multi-turn continuity work. Survives
        # bot restarts; dropped by send_control("/clear"). See the module
        # docstring for why it cannot be re-derived (resume takes no -s/-C).
        self._thread_file = self.runtime_dir / "thread_id"
        # Raw JSONL, append-only: forensic material, and the liveness signal
        # is_working() watches (this engine's analog of pane.log growth).
        self._event_log = self.runtime_dir / "codex.log"
        # One structured line per turn — what capture_text() returns. See
        # capture_text()'s docstring for why the vendor's prose is NOT in it.
        self._turn_journal = self.runtime_dir / "turns.log"
        # Last successfully delivered reply, for /resend.
        self._last_reply = self.runtime_dir / "last_reply"
        # Replies that completed but that no ask() could hand back.
        self._orphans = self.runtime_dir / "orphans"
        # PID of the live exec process — the cross-process half of
        # is_pane_turn_active() (the watchdog is a different process).
        self._pid_file = self.runtime_dir / "turn.pid"
        # send_control("/model X") override, applied on top of self.model.
        self._model_file = self.runtime_dir / "model"

        # is_working() change tracking (see that method).
        self._last_log_size = -1
        self._frozen_since: float | None = None

    # ── small file helpers ───────────────────────────────────────────

    def _atomic_write(self, target: Path, payload: str) -> bool:
        """tmp-then-rename, same rationale as ClaudeSession._atomic_write:
        a bare write_text is a truncate-then-write, and a process killed
        mid-write (delivery_guard's SIGKILL) would leave a torn file."""
        try:
            tmp = target.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(payload)
            os.replace(tmp, target)
            return True
        except OSError as exc:
            logger.error("could not write %s: %s", target, exc)
            return False

    @staticmethod
    def _read_text(path: Path) -> str | None:
        try:
            value = path.read_text().strip()
        except OSError:
            return None
        return value or None

    def _log_size(self) -> int:
        try:
            return self._event_log.stat().st_size
        except OSError:
            return 0

    def _append_event_line(self, line: str) -> None:
        try:
            if self._log_size() > _EVENT_LOG_CAP_BYTES:
                # Truncate rather than rotate: this file is forensic, never
                # authoritative — nothing reads it for state.
                self._event_log.write_text("")
            with self._event_log.open("a") as fh:
                fh.write(line if line.endswith("\n") else line + "\n")
        except OSError as exc:
            logger.warning("could not append to %s: %s", self._event_log, exc)

    def _journal_turn(self, **fields: object) -> None:
        parts = " ".join(f"{k}={v}" for k, v in fields.items())
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            with self._turn_journal.open("a") as fh:
                fh.write(f"{stamp} {parts}\n")
        except OSError as exc:
            logger.warning("could not append to %s: %s", self._turn_journal, exc)

    # ── the turn lock ────────────────────────────────────────────────

    def _try_lock(self) -> int | None:
        """Non-blocking acquire; returns the held fd, or None if contended."""
        fd = os.open(self._turn_lock, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        return fd

    @staticmethod
    def _unlock(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def is_turn_active(self) -> bool:
        """True iff a turn is in flight (the turn lock is held).

        Lock-based, exactly like ``ClaudeSession.is_turn_active`` — and,
        unlike the tmux engine, the lock genuinely cannot outlive its holder
        here: flock is released by the kernel when the process dies, and a
        turn is one process. There is no "stale busy pane" class of bug for
        this engine.
        """
        fd = self._try_lock()
        if fd is None:
            return True
        self._unlock(fd)
        return False

    def _acquire_turn(self, log_id: str) -> tuple[int | None, AskResult | None]:
        """Bounded acquire of the turn lock.

        Never blocks forever (the Protocol's hard requirement). On giving up
        it classifies the holder the same way ``ClaudeSession`` classifies a
        busy pane, using this engine's honest progress signal — growth of the
        raw event log:

        * the log grew during the wait ⇒ ``busy_active`` (a live, working
          turn; ``ask_health`` scores this as neutral, not a delivery failure)
        * the log never grew ⇒ ``busy`` (wedged; counts as a failure, which
          is what keeps ``delivery_guard``'s restart backstop armed)
        """
        fd = self._try_lock()
        if fd is not None:
            return fd, None
        start = self._clock()
        deadline = start + self._busy_wait_budget
        size_at_start = self._log_size()
        grew = False
        while self._clock() < deadline:
            self._sleep(self._poll_interval)
            if self._log_size() > size_at_start:
                grew = True
            fd = self._try_lock()
            if fd is not None:
                return fd, None
        waited = self._clock() - start
        if grew:
            logger.info(
                "codex turn lock held by a live, progressing turn — not "
                "starting %s on top of it",
                log_id,
            )
            return None, AskResult(
                "busy_active",
                detail="another codex turn is in flight and progressing",
                busy_seconds=waited,
            )
        logger.error(
            "codex turn lock held with no stream progress for %.0fs — "
            "refusing to start %s",
            waited,
            log_id,
        )
        return None, AskResult(
            "busy",
            detail="another codex turn holds the lock and is not progressing",
            busy_seconds=waited,
        )

    # ── lifecycle ────────────────────────────────────────────────────

    def ensure_session(self) -> None:
        """Idempotent readiness check.

        There is no long-lived process to start — a "session" here is the
        persisted ``thread_id`` plus a runtime dir. What CAN be wrong at this
        point is the CLI being missing, so that is what is checked, loudly:
        the tmux engine refuses to start a personality-less brain, this one
        refuses to pretend a turn is possible with no ``codex`` binary.
        """
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which(self.codex_bin) is None:
            raise RuntimeError(
                f"codex binary {self.codex_bin!r} not found on PATH — "
                "refusing to start the codex engine"
            )
        if self.instructions_file is not None and not self.instructions_file.exists():
            raise RuntimeError(
                f"codex instructions file missing: {self.instructions_file} — "
                "refusing to start a personality-less brain"
            )

    def is_healthy(self) -> bool:
        """Cheap liveness probe.

        ``ClaudeSession`` asks tmux whether the session exists. There is no
        equivalent object here, so the honest cheapest question is "could a
        turn run at all right now" — the CLI is invocable and the runtime dir
        is writable. It deliberately does NOT call the network: this is
        polled by the watchdog and must stay free.
        """
        if shutil.which(self.codex_bin) is None:
            return False
        return os.access(self.runtime_dir, os.W_OK)

    def _live_pid(self) -> int | None:
        raw = self._read_text(self._pid_file)
        if raw is None:
            return None
        try:
            pid = int(raw)
        except ValueError:
            return None
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return None
        return pid

    def is_pane_turn_active(self) -> bool:
        """True iff the exec process is alive, whoever holds the lock.

        The Protocol names this after tmux for continuity; for a
        process-per-turn engine it is literally "the exec process is alive",
        read from the pid file so a DIFFERENT process (the watchdog, the cron
        runner) can ask the question too.
        """
        return self._live_pid() is not None

    def is_working(self) -> bool:
        """Change-aware health probe: is there a turn making progress?

        The tmux engine has to distinguish a live turn from a frozen frame
        showing a stale spinner (``STATIC_TRUST_WINDOW`` and friends). Here
        the two signals are unambiguous and cheap: the process is alive, and
        the JSONL event log is growing. A live process whose stream has
        produced nothing for ``stall_timeout`` is reported as NOT working —
        which is exactly the wedge the watchdog exists to notice.
        """
        if not self.is_pane_turn_active():
            self._last_log_size = -1
            self._frozen_since = None
            return False
        size = self._log_size()
        now = self._clock()
        if self._last_log_size < 0 or size > self._last_log_size:
            self._frozen_since = None
        elif self._frozen_since is None:
            self._frozen_since = now
        self._last_log_size = size
        if self._frozen_since is not None and (
            now - self._frozen_since >= self._stall_timeout
        ):
            return False
        return True

    def _kill_stray(self) -> None:
        pid = self._live_pid()
        if pid is None:
            self._pid_file.unlink(missing_ok=True)
            return
        logger.warning("codex: killing stray exec process pid=%s", pid)
        for sig in (signal.SIGINT, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except OSError:
                break
            deadline = self._clock() + self._interrupt_grace
            while self._clock() < deadline:
                if self._live_pid() is None:
                    break
                self._sleep(self._poll_interval)
            if self._live_pid() is None:
                break
        self._pid_file.unlink(missing_ok=True)

    def force_recover(self) -> bool:
        """Watchdog entry point. False ⇒ a live turn holds the lock.

        DELIBERATELY LESS DESTRUCTIVE THAN THE CLAUDE EQUIVALENT, and this is
        a judgement call worth stating. ``ClaudeSession.force_recover`` does
        ``tmux kill-session``, which destroys the conversation along with the
        wedge, because for a TUI the two are the same object. For Codex they
        are not: the wedge is one exec PROCESS, while the conversation is a
        durable ``thread_id`` on disk. Spike step 5 proved the split empirically
        — SIGINT mid-turn, then ``resume`` the same thread: rc=0, same
        thread_id, partial work intact. So recovery kills the process and
        keeps the thread. Dropping context here would be gratuitous damage,
        not caution. ``send_control("/clear")`` remains the explicit way to
        throw the conversation away.
        """
        fd = self._try_lock()
        if fd is None:
            return False
        try:
            self._kill_stray()
            self._inflight.unlink(missing_ok=True)
            return True
        finally:
            self._unlock(fd)

    def kill(self) -> None:
        """Tear down whatever is running (CLI/teardown use).

        Does NOT wait politely for the lock: "kill" that blocks behind the
        thing it is meant to kill would be useless. The thread survives, same
        reasoning as ``force_recover``.
        """
        self._kill_stray()
        self._inflight.unlink(missing_ok=True)

    # ── control / steering ───────────────────────────────────────────

    def interrupt(self) -> None:
        """Stop the current turn — SIGINT to the exec process.

        Spike step 5, measured: the process exits on its own (rc=1), no
        ``turn.completed`` is emitted (so "interrupted" is distinguishable
        from "finished" by the same signal that distinguishes an error), and
        the thread is NOT damaged — the next ``resume`` returns rc=0 with the
        same thread_id and the partial work sitting in the transcript.

        Takes no lock, mirroring ``ClaudeSession.interrupt``: the lock is
        held by the very turn being interrupted.
        """
        pid = self._live_pid()
        if pid is None:
            logger.info("codex interrupt: no live turn")
            return
        try:
            os.kill(pid, signal.SIGINT)
            logger.info("codex interrupt: SIGINT sent to pid=%s", pid)
        except OSError as exc:
            logger.warning("codex interrupt: could not signal pid=%s: %s", pid, exc)

    def is_steerable_turn(self) -> bool:
        """Always False — NO PARITY, see the module docstring.

        A ``codex exec`` process reads its prompt once at start-up and its
        stdin is DEVNULL by necessity (an inherited stdin makes the CLI wait
        on "Reading additional input from stdin..." forever). There is no
        mid-turn input channel to be steerable *into*.

        Returning False is the honest answer AND the one with the best
        behavior: ``bot/handlers/chat.py`` already handles it by telling the
        user to resend in a few minutes, so their text is never silently
        swallowed — it just does not get injected into the running turn.
        """
        return False

    def steer(self, text: str) -> None:
        """No-op with a loud log — unreachable through the bot.

        Kept as a real method (the Protocol requires the name) but it cannot
        do what its contract says on this engine. It is only reachable if
        something calls it WITHOUT consulting ``is_steerable_turn()`` first;
        rather than drop the text, log it at ERROR so it is recoverable from
        the journal, and say plainly that nothing was injected.
        """
        logger.error(
            "codex engine cannot steer a live turn (no mid-turn stdin "
            "channel) — text NOT delivered to the model: %r",
            text,
        )

    def send_control(self, text: str) -> None:
        """Client-side control command. Semantics are engine-specific.

        Claude Code's ``/clear`` and ``/model`` are TUI commands typed into a
        live client. Codex has no client to type into, so each is mapped to
        the operation that produces the SAME OBSERVABLE EFFECT, and anything
        without such an operation is refused loudly instead of faked:

        ``/clear`` → forget the persisted ``thread_id``. The next ``ask()``
            starts a brand-new thread, which is exactly Codex's own model of
            "fresh context" (spike step 2: omitting ``resume`` yields a new
            thread_id; there is no in-place clear). This is also precisely
            what ``cron_runner`` wants when it clears after every job, and it
            is cheaper and more reliable than Claude's version — no session-id
            resync is needed, because the id is ours and we just deleted it.

        ``/model <name>`` → persist a model override applied via ``-m`` on
            subsequent turns. ``/model`` with no argument is a picker in the
            TUI; there is no non-interactive equivalent, so it is refused.
            NOTE: the account's model catalog is *account-scoped* (spike step
            6 hit "model is not supported when using Codex with a ChatGPT
            account"), so an invalid name surfaces as a turn error, not here.

        anything else → logged as unsupported and ignored. Control commands
            produce no model turn either way, so there is nothing to return;
            the alternative (silently doing nothing) is what this branch
            exists to avoid.
        """
        command = text.strip()
        if command == "/clear":
            old = self._read_text(self._thread_file)
            self._thread_file.unlink(missing_ok=True)
            logger.info("codex /clear: dropped thread %s — next turn starts fresh", old)
            return
        if command.startswith("/model"):
            parts = command.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                self._atomic_write(self._model_file, parts[1].strip() + "\n")
                logger.info("codex /model: model override set to %s", parts[1].strip())
            else:
                logger.warning(
                    "codex /model with no argument is interactive-only — "
                    "unsupported on the codex engine; pass a model name"
                )
            return
        logger.warning(
            "codex engine ignores control command %r — no equivalent exists", command
        )

    # ── delivery salvage ─────────────────────────────────────────────

    def capture_text(self) -> str:
        """This engine's output surface: the structured per-turn journal.

        NOT the raw ``--json`` stream, and that is a deliberate safety
        decision rather than a convenience one. ``cron_runner._limit_recovery``
        feeds ``capture_text()`` straight into ``tmux_parse.classify_state``,
        whose rate-limit regex matches the substring "you've hit your ...
        limit" — which is *verbatim* the user-facing text Codex emits when a
        limit is reached (spike step 1). Returning raw vendor prose would let
        a Codex error message trip a Claude-shaped classifier and make the
        cron runner fire ``send_control("/clear")``, silently throwing away
        the thread. So this returns only tokens this driver itself writes
        (statuses from our own vocabulary, ids, counts, durations), which
        cannot match those signatures by construction — see
        ``tests/test_codex_driver.py::test_capture_text_never_trips_tmux_classifiers``.

        The raw stream is still kept, untouched, in ``runtime_dir/codex.log``
        for forensics; nothing reads it for state.
        """
        try:
            size = self._turn_journal.stat().st_size
            with self._turn_journal.open() as fh:
                if size > _CAPTURE_TAIL_BYTES:
                    fh.seek(size - _CAPTURE_TAIL_BYTES)
                    fh.readline()  # drop the partial first line
                body = fh.read()
        except OSError:
            body = ""
        thread = self._read_text(self._thread_file) or "none"
        head = (
            f"engine=codex session={self.session_name} thread={thread} "
            f"turn_active={self.is_pane_turn_active()}\n"
        )
        return head + body

    def last_reply_for_resend(self) -> tuple[str, str | None]:
        """``(status, body)`` for ``/resend``. Read-only, never waits.

        Statuses match ``ClaudeSession``'s vocabulary so ``resend.py`` needs
        no engine awareness. ``"no_markers"`` and ``"unclosed"`` cannot occur
        here — they describe marker-scraping failures that do not exist on
        this engine — so this only ever returns ``unavailable`` / ``empty`` /
        ``in_progress`` / ``ready``.

        Reads the reply this driver itself recorded rather than re-parsing a
        transcript: unlike the tmux engine, the reply we delivered and the
        reply the model produced are the same string by construction, so
        there is nothing to reconcile.
        """
        if self.is_turn_active() or self.is_pane_turn_active():
            return "in_progress", None
        body = None
        try:
            body = self._last_reply.read_text()
        except OSError:
            body = None
        if body:
            return "ready", body
        if self._read_text(self._thread_file) is None:
            return "unavailable", None
        return "empty", None

    def pop_orphan_replies(self) -> list[str]:
        """Completed replies nobody's ``ask()`` could take, oldest first.

        Structurally almost impossible on this engine — a reply is returned
        by the very ``ask()`` that started the process — with exactly one
        real exception, which is why the queue exists: ``ask()`` hits its hard
        deadline, sends SIGINT, and while draining the dying process observes
        a ``turn.completed`` that arrived just too late for the caller. That
        answer is real and finished, and would otherwise be lost; it is queued
        here and the watchdog delivers it on its next idle tick.

        Consumed exactly once. Returns nothing while a turn is in flight, for
        the same reason ``ClaudeSession`` does not: no reading state out from
        under a live turn.
        """
        if self.is_turn_active() or self.is_pane_turn_active():
            return []
        fd = os.open(self._state_lock, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                raw = self._orphans.read_text()
            except OSError:
                return []
            if not raw.strip():
                return []
            try:
                items = json.loads(raw)
            except json.JSONDecodeError:
                logger.error("codex: orphan queue is corrupt — dropping it")
                self._atomic_write(self._orphans, "")
                return []
            self._atomic_write(self._orphans, "")
            return [str(x) for x in items if str(x).strip()]
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _queue_orphan(self, reply: str) -> None:
        fd = os.open(self._state_lock, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                items = json.loads(self._orphans.read_text() or "[]")
            except (OSError, json.JSONDecodeError):
                items = []
            items.append(reply)
            self._atomic_write(self._orphans, json.dumps(items, ensure_ascii=False))
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # ── the turn ─────────────────────────────────────────────────────

    def _effective_model(self) -> str | None:
        return self._read_text(self._model_file) or self.model

    def _argv(self, thread_id: str | None, prompt: str) -> list[str]:
        """Build the exec command line.

        The ``resume`` branch carries NEITHER ``-s`` nor ``-C``: the CLI
        rejects them outright ("error: unexpected argument"), because the
        sandbox and working root are properties of the thread, fixed when it
        was created. This is the single most load-bearing CLI fact in this
        module — see the module docstring.
        """
        argv = [self.codex_bin, "exec"]
        if thread_id:
            argv += ["resume", thread_id, "--json", "--skip-git-repo-check"]
        else:
            argv += [
                "--json",
                "-s",
                self.sandbox,
                "--skip-git-repo-check",
                "-C",
                str(self.work_dir),
            ]
        model = self._effective_model()
        if model:
            argv += ["-m", model]
        # "--" stops clap from parsing a prompt that happens to start with
        # "-" as a flag (e.g. "- надо купить молока") — without it such a
        # message is rejected with rc=2 "unexpected argument". Found live
        # during independent review: the persona preamble masks this on the
        # first turn of a thread only, so every resumed turn was exposed.
        argv.append("--")
        argv.append(prompt)
        return argv

    def _compose_prompt(self, prompt: str, *, fresh_thread: bool) -> str:
        """Prepend the persona to the FIRST prompt of a thread.

        Codex's native mechanism is ``AGENTS.md`` in the working root, and
        the spike proved it works well (step 4: 12/12 masculine grammatical
        gender with no reminder in the prompt, 12/12 allowed-tags-only). This
        driver deliberately does NOT write that file: ``work_dir`` is the
        live vault, a git repo this bot commits from, and dropping a
        generated file into it would be a production-visible side effect for
        an engine nobody has switched on yet.

        Injecting the instructions as the preamble of the thread's first turn
        is equivalent in effect and free in cost: the thread persists, so
        every later ``resume`` carries the persona in its replayed context,
        and the spike measured that replay as ~86% cache hits. A vault that
        DOES have its own ``AGENTS.md`` still gets it — Codex picks that up
        natively, in addition, with no help from here.
        """
        if not fresh_thread or self.instructions_file is None:
            return prompt
        try:
            persona = self.instructions_file.read_text().strip()
        except OSError as exc:
            logger.error(
                "could not read codex instructions %s: %s — sending the "
                "prompt without a persona",
                self.instructions_file,
                exc,
            )
            return prompt
        return (
            f"{persona}\n\n"
            "--- end of standing instructions; the user's message follows ---\n\n"
            f"{prompt}"
        )

    def _rollout_files(self, thread_id: str) -> list[Path]:
        home = self.codex_home or Path(
            os.environ.get("CODEX_HOME") or (Path.home() / ".codex")
        )
        try:
            return sorted((home / "sessions").rglob(f"rollout-*{thread_id}.jsonl"))
        except OSError:
            return []

    def _rollout_rate_limited(self, thread_id: str | None) -> bool:
        """Is the account's limit actually reached, per the rollout file?

        The spike's explicit instruction to phase 2: rate-limit telemetry is
        NOT in the ``--json`` stream, only in the thread's rollout JSONL as a
        ``token_count`` event carrying ``rate_limits.rate_limit_reached_type``
        (verified again here — the field is present and null while healthy).
        So a generic error is cross-checked against it before being called an
        ``error``, which is what keeps ``rate_limited`` from being reported as
        a delivery failure by ``ask_health``.
        """
        if not thread_id:
            return False
        reached = None
        for path in self._rollout_files(thread_id):
            try:
                with path.open() as fh:
                    for line in fh:
                        if '"rate_limits"' not in line:
                            continue
                        try:
                            payload = json.loads(line).get("payload", {})
                        except json.JSONDecodeError:
                            continue
                        limits = payload.get("rate_limits")
                        if isinstance(limits, dict):
                            reached = limits.get("rate_limit_reached_type")
            except OSError:
                continue
        return bool(reached)

    @staticmethod
    def _classify_error(message: str) -> str | None:
        low = (message or "").lower()
        if any(m in low for m in _LOGGED_OUT_MARKERS):
            return "logged_out"
        if any(m in low for m in _RATE_LIMIT_MARKERS):
            return "rate_limited"
        return None

    def _handle_event(self, event: dict, outcome: _TurnOutcome) -> None:
        kind = event.get("type")
        if kind == "thread.started":
            tid = event.get("thread_id")
            if tid and tid != outcome.thread_id:
                outcome.thread_id = tid
                # Persisted the moment it is known, not at the end of the
                # turn: if this process dies mid-turn, the thread must still
                # be resumable rather than orphaned in ~/.codex/sessions.
                self._atomic_write(self._thread_file, f"{tid}\n")
        elif kind == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    outcome.reply = text
        elif kind == "turn.completed":
            outcome.completed = True
        elif kind == "turn.failed":
            err = event.get("error") or {}
            outcome.error = str(err.get("message") or err or "turn failed")
        elif kind == "error":
            outcome.error = str(event.get("message") or "error")

    @staticmethod
    def _strip_markers(reply: str) -> str:
        """Remove a ``<<<R:id>>>``/``<<<E:id>>>`` pair if the model emitted one.

        This engine does not need the marker contract — the stream tells us
        where the turn ends — and ``ask()`` never asks for markers. But the
        SHARED vault skills and any operator-written ``AGENTS.md`` may still
        instruct the model to emit them (they were written for the Claude
        engine and are also read by the other engine's session, so they must not be
        forked). Stripping is defensive, cheap, and keeps the delivered text
        clean either way.
        """
        lines = [ln for ln in reply.splitlines() if not _is_marker_line(ln)]
        return "\n".join(lines).strip() or reply.strip()

    def ask(
        self,
        prompt: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        request_id: str | None = None,
        wrap: bool = True,
    ) -> AskResult:
        """Run one turn as a ``codex exec`` process and map its outcome.

        Never raises and never blocks forever, same contract as
        ``ClaudeSession.ask``. Status mapping, all of it:

        ``ok``           ``turn.completed`` seen and an ``agent_message``
                         carried text.
        ``error``        process failed / ``turn.failed`` with an
                         unclassifiable message / completed with no reply at
                         all / the CLI could not be launched.
        ``timeout``      the hard ``timeout`` elapsed, OR no stream event for
                         ``stall_timeout`` (this engine's wedge signal — the
                         plan's mapping, and the honest one: nothing was
                         delivered and nothing crashed).
        ``rate_limited`` error text matches the limit signatures, or the
                         thread's rollout file says the limit was reached.
        ``logged_out``   error text matches the auth signatures.
        ``busy`` /       the turn lock is held by another turn — see
        ``busy_active``  ``_acquire_turn``.

        ``wrap`` IS ACCEPTED AND DELIBERATELY DOES NOT CHANGE THE PROMPT.
        On the tmux engine it appends the ``<<<R:id>>>`` marker instruction
        because the reply has to be found in a screen scrape. Here the reply
        arrives as a discrete ``agent_message`` item, so asking the model to
        wrap it would add a failure mode (a dropped closing marker) to buy
        nothing. The parameter stays in the signature because it is part of
        the shared Protocol, and callers that pass ``wrap=False`` (verbatim
        prompts) get exactly what they expect: the prompt, unmodified.
        """
        rid = self._rid_factory()
        log_id = request_id or rid
        fd, busy = self._acquire_turn(log_id)
        if fd is None:
            return busy if busy is not None else AskResult("error", detail="no lock")
        try:
            self._inflight.write_text(f"{log_id}\n{self._clock()}\n")
            try:
                self.ensure_session()
            except Exception as exc:  # noqa: BLE001 — must never escape ask()
                logger.error("codex ensure_session failed for %s: %s", log_id, exc)
                return AskResult("error", detail=f"session start failed: {exc}")
            return self._run_turn(prompt, log_id=log_id, timeout=timeout)
        finally:
            self._inflight.unlink(missing_ok=True)
            self._pid_file.unlink(missing_ok=True)
            self._unlock(fd)

    def _run_turn(self, prompt: str, *, log_id: str, timeout: float) -> AskResult:
        thread_id = self._read_text(self._thread_file)
        payload = self._compose_prompt(prompt, fresh_thread=thread_id is None)
        argv = self._argv(thread_id, payload)
        started = self._clock()
        try:
            proc = self._popen(
                argv,
                cwd=str(self.work_dir),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            logger.error("codex: could not launch %s: %s", argv[0], exc)
            self._journal_turn(rid=log_id, status="error", reason="spawn_failed")
            return AskResult("error", detail=f"could not launch codex: {exc}")

        pid = getattr(proc, "pid", None)
        if pid is not None:
            self._atomic_write(self._pid_file, f"{pid}\n")

        outcome = _TurnOutcome(thread_id=thread_id)
        reader = _EventReader(proc.stdout, self._append_event_line)
        deadline = started + timeout
        last_event = started
        verdict: str | None = None  # set when a deadline forces the end

        while True:
            line = reader.next_line(self._poll_interval)
            if line is None:  # EOF: the process closed its stdout
                break
            if line is not _EventReader.NOTHING:
                last_event = self._clock()
                text = str(line).strip()
                if text.startswith("{"):
                    try:
                        event = json.loads(text)
                    except json.JSONDecodeError:
                        logger.debug("codex: non-JSON stdout line: %r", text[:200])
                    else:
                        outcome.events += 1
                        self._handle_event(event, outcome)
                        if outcome.completed:
                            break
                continue
            # No event this tick — the only place deadlines are evaluated.
            now = self._clock()
            if now >= deadline:
                verdict = "timeout"
                logger.warning(
                    "codex turn %s exceeded the hard timeout of %ss — interrupting",
                    log_id,
                    timeout,
                )
                break
            if now - last_event > self._stall_timeout:
                verdict = "stalled"
                logger.warning(
                    "codex turn %s produced no stream event for %ss — interrupting",
                    log_id,
                    self._stall_timeout,
                )
                break

        if verdict is not None:
            # Drain whatever the dying process still emits: a turn.completed
            # that lands here is a REAL, finished answer that simply arrived
            # too late for this caller — see pop_orphan_replies().
            self._terminate(proc)
            self._drain_after_kill(reader, outcome)
        else:
            self._reap(proc)

        stderr_tail = self._read_stderr(proc)
        rc = getattr(proc, "returncode", None)
        elapsed = self._clock() - started
        return self._finish(
            outcome,
            verdict=verdict,
            rc=rc,
            stderr_tail=stderr_tail,
            log_id=log_id,
            elapsed=elapsed,
            timeout=timeout,
        )

    def _terminate(self, proc) -> None:
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:  # noqa: BLE001 — already dead is fine
            logger.debug("codex: SIGINT failed (process already gone)", exc_info=True)
        deadline = self._clock() + self._interrupt_grace
        while self._clock() < deadline:
            if proc.poll() is not None:
                return
            self._sleep(self._poll_interval)
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            logger.debug("codex: kill failed", exc_info=True)
        # Plain wait, deliberately NOT self._reap(): _reap escalates back to
        # _terminate for a process that outlives its turn, and the two calling
        # each other is a loop with no exit condition. SIGKILL has been sent;
        # this is only collecting the corpse.
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 — best effort; rc may stay None
            logger.debug("codex: could not reap killed process", exc_info=True)

    def _drain_after_kill(self, reader: _EventReader, outcome: _TurnOutcome) -> None:
        """Read what the dying process still had to say.

        Bounded twice over — by EOF (the normal end: the process exits and its
        stdout closes) and by ``_DRAIN_MAX_IDLE`` consecutive empty reads, so a
        process that ignores SIGINT and keeps streaming cannot hold ``ask()``
        here forever. Worth doing at all because a ``turn.completed`` landing
        in this window is a REAL finished answer that would otherwise be lost;
        see ``pop_orphan_replies``.
        """
        idle = 0
        while idle < _DRAIN_MAX_IDLE:
            line = reader.next_line(_DRAIN_READ_TIMEOUT)
            if line is None:
                return  # EOF
            if line is _EventReader.NOTHING:
                idle += 1
                continue
            idle = 0
            text = str(line).strip()
            if not text.startswith("{"):
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue
            outcome.events += 1
            self._handle_event(event, outcome)

    def _reap(self, proc) -> None:
        """Wait out a process that has already said everything it is going to.

        Reached on the SUCCESS path, where we break out of the read loop the
        instant ``turn.completed`` arrives — a moment before the CLI actually
        exits. It normally exits within milliseconds. A process that does NOT
        is a leak (and would keep the pid file's promise alive after ``ask()``
        has cleared it), so it gets terminated rather than left behind: the
        turn is over either way, there is nothing left to lose by killing it.
        """
        try:
            proc.wait(timeout=5)
            return
        except Exception:  # noqa: BLE001 — still running, or a fake without wait
            logger.debug("codex: process did not exit on its own", exc_info=True)
        if proc.poll() is None:
            logger.warning("codex: process outlived its completed turn — killing it")
            self._terminate(proc)

    @staticmethod
    def _read_stderr(proc) -> str:
        """Drain stderr, but ONLY from a process known to have exited.

        ``read()`` on a live process's pipe blocks until EOF, which would be a
        way for ``ask()`` to hang forever after it had already decided the
        outcome — precisely what the Protocol forbids. Every caller reaches
        here after ``_reap``/``_terminate``, so a still-None returncode means
        something pathological is going on and the stderr tail (a diagnostic
        nicety) is simply not worth waiting for.
        """
        if getattr(proc, "returncode", None) is None:
            return ""
        stream = getattr(proc, "stderr", None)
        if stream is None:
            return ""
        try:
            return (stream.read() or "")[-2000:]
        except Exception:  # noqa: BLE001
            return ""

    def _finish(
        self,
        outcome: _TurnOutcome,
        *,
        verdict: str | None,
        rc: int | None,
        stderr_tail: str,
        log_id: str,
        elapsed: float,
        timeout: float,
    ) -> AskResult:
        def journal(status: str, **extra: object) -> None:
            self._journal_turn(
                rid=log_id,
                status=status,
                events=outcome.events,
                rc=rc,
                dur=f"{elapsed:.1f}s",
                **extra,
            )

        if outcome.completed and outcome.reply:
            reply = self._strip_markers(outcome.reply)
            if verdict is not None:
                # Finished, but after this caller had already given up on it.
                # Real answer, no live caller: queue it for the watchdog
                # instead of dropping it on the floor.
                self._queue_orphan(reply)
                journal("timeout", late_reply="queued")
                logger.warning(
                    "codex turn %s completed while being interrupted — reply "
                    "queued as an orphan (len=%d)",
                    log_id,
                    len(reply),
                )
                return AskResult(
                    "timeout", detail=f"reply arrived after the {verdict} deadline"
                )
            self._atomic_write(self._last_reply, reply)
            journal("ok", len=len(reply))
            return AskResult("ok", reply=reply)

        if verdict == "timeout":
            journal("timeout", reason="hard_deadline")
            return AskResult("timeout", detail=f"no reply in {timeout}s")
        if verdict == "stalled":
            journal("timeout", reason="stalled")
            return AskResult(
                "timeout",
                detail=f"no codex stream event for {self._stall_timeout}s",
            )

        if outcome.error is None and not outcome.completed and outcome.events:
            # CANCELLED, not broken — and the stream says so unambiguously.
            # A genuine API failure emits `{"type":"error"}` AND
            # `{"type":"turn.failed"}` before exiting (spike step 1); an
            # externally SIGINT'ed turn emits NEITHER and simply stops with
            # rc=1 (spike step 5, reproduced live here while building this
            # driver: 2 events — thread.started, turn.started — then nothing).
            # So "the stream started, produced no error, and never completed"
            # is precisely the interrupt shape.
            #
            # Reported as "error" rather than a new status on purpose. The
            # shared vocabulary has no "cancelled", and inventing one would
            # silently change how ask_health and delivery_guard score turns
            # they have never seen before. This also keeps parity with the
            # tmux engine, whose own interrupt path likewise lands in an
            # error-class status (ask()'s stall branch returns
            # AskResult("error", detail="session stalled")). Only the DETAIL
            # is made honest, so the journal says "cancelled" instead of
            # blaming a failure that never happened.
            journal("error", reason="cancelled")
            logger.info(
                "codex turn %s ended without completing and without an error "
                "event (rc=%s) — interrupted or killed externally",
                log_id,
                rc,
            )
            return AskResult(
                "error",
                detail="turn was interrupted before it produced a reply "
                "(codex reported no error)",
            )

        message = outcome.error or stderr_tail or f"codex exited with rc={rc}"
        status = self._classify_error(message)
        if status is None and self._rollout_rate_limited(outcome.thread_id):
            # The stream said only "something failed"; the rollout telemetry
            # is the ONLY place that can say "because the limit was reached".
            status = "rate_limited"
        if status == "rate_limited":
            journal("rate_limited")
            logger.warning("codex turn %s hit a usage limit: %s", log_id, message[:300])
            return AskResult("rate_limited", detail=message[:500])
        if status == "logged_out":
            journal("logged_out")
            logger.error("codex turn %s: session is logged out", log_id)
            return AskResult("logged_out", detail=message[:500])
        if outcome.completed and not outcome.reply:
            # turn.completed with no agent_message: the model finished
            # without saying anything. Not "ok" — there is nothing to
            # deliver, and reporting ok would hand the user an empty message.
            journal("error", reason="empty_reply")
            return AskResult("error", detail="codex turn completed with no reply")
        journal("error", reason="turn_failed")
        logger.error("codex turn %s failed (rc=%s): %s", log_id, rc, message[:500])
        return AskResult("error", detail=message[:500])

    # ── compatibility shims (NOT part of EngineDriver) ───────────────
    #
    # watchdog.py and cron_runner.py reach for a handful of Claude-shaped
    # methods that engine.py deliberately kept OUT of the Protocol because
    # they are tmux concepts. They are implemented here as conservative
    # no-ops so that a codex-backed session degrades to "the watchdog finds
    # nothing to do" instead of crash-looping on AttributeError. A watchdog
    # that actually understands this engine is explicitly out of the
    # compressed phase-2 scope (agent-infra-backlog item 26).

    def current_state(self):
        """Always ``PaneState.READY`` — the tmux vocabulary the watchdog speaks.

        Every other member of that enum describes a long-lived TUI parked on a
        screen: TRUST_PROMPT and BYPASS_PROMPT are startup dialogs, STARTING is
        a welcome box, RATE_LIMITED and LOGGED_OUT are banners the client sits
        on waiting for a keystroke. None of them can exist here — a limit or a
        logout on this engine is a FAILED TURN, reported through ``AskResult``
        where ``ask_health`` already accounts for it, and there is no screen to
        park on afterwards. UNKNOWN would be worse than READY, not more
        cautious: UNKNOWN is outside ``watchdog._SERVICEABLE``, so it would arm
        ``_is_hung`` against a session whose only recovery path is
        ``force_recover``, on the strength of a signal this engine cannot
        produce. READY keeps the watchdog on its idle branch, which is the one
        that actually does something useful here — delivering queued orphan
        replies (see ``pop_orphan_replies``). Liveness, when the watchdog wants
        it, comes from ``is_working()``, which this engine answers precisely.
        """
        from d_brain.services.tmux_parse import PaneState

        return PaneState.READY

    def nudge(self, text: str = "Continue") -> bool:
        """Always False — nothing to wake.

        The Claude version exists because that CLI parks at a subscription
        banner and waits for a keystroke. A ``codex exec`` process either
        runs or does not exist; there is no parked state, so there is nothing
        a nudge could do. Returning False is the honest answer and is what
        the watchdog already treats as "the nudge did not happen".
        """
        return False

    def clear(self) -> None:
        """Thin wrapper over ``send_control('/clear')`` — see there."""
        self.send_control("/clear")

    def current_transcript_path(self) -> Path | None:
        """The thread's rollout JSONL, Codex's nearest thing to a transcript.

        Not the same object as Claude Code's ``~/.claude/projects/**.jsonl``
        (different schema entirely), so callers that PARSE a Claude
        transcript will not get what they expect. It is returned anyway
        because the honest answer to "where does this session's history live"
        is this file, and because ``/resend`` does not go through it on this
        engine (see ``last_reply_for_resend``).
        """
        thread_id = self._read_text(self._thread_file)
        if thread_id is None:
            return None
        files = self._rollout_files(thread_id)
        return files[-1] if files else None


def _is_marker_line(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith("<<<R:") or stripped.startswith("<<<E:")
    ) and stripped.endswith(">>>")
