"""Engine seam: the structural contract every brain backend must satisfy.

Phase 1 of the engine implementation plan.
The ONLY thing this module adds today is a *name* for the surface the bot
already calls on ``ClaudeSession``: a ``typing.Protocol``. Nothing here is
imported by the live reply path, nothing here changes behavior. It exists so
that ``runtime.py`` has a place to branch on an engine flag in phase 2
without anyone having to touch ``claude_session.py``.

Why a Protocol and not an ABC
-----------------------------
``ClaudeSession`` must not be edited at all — it is delivery-critical code
hardened by real incidents (agent-infra-backlog items 10, 11, 13, 21-24), and
adding a base class to it would be an edit. Structural typing needs no
inheritance and no wrapper: ``ClaudeSession`` conforms *as it is written
today*, which ``tests/test_engine.py`` verifies signature-by-signature rather
than asserting it in prose. The future ``CodexExecDriver`` (phase 2) is a
separate class that conforms the same way.

Deliberately NOT part of this Protocol
--------------------------------------
The engine-neutral contract is narrower than ``ClaudeSession``'s full public
surface. Left out on purpose, per the plan's "честный грязный вырез":

* ``current_state() -> PaneState`` — the return type is a tmux concept
  (``tmux_parse.PaneState``). ``watchdog.py`` and ``cron_runner.py`` import
  ``PaneState``/``classify_state`` directly; making that engine-agnostic is
  explicitly deferred to phase 2 (and only if Codex proves out). A Codex
  session simply will not run the tmux watchdog.
* ``nudge()``, ``clear()``, ``current_transcript_path()`` — Claude-Code-shaped
  operations (wake a session parked on a subscription banner, the client-side
  ``/clear`` with session-id resync, the JSONL transcript path). Their Codex
  equivalents have different semantics; phase 2 decides whether they earn a
  place in the shared contract or stay backend-specific.
* Attributes (``session_name``, ``work_dir``, ``runtime_dir``, …) — read by
  the doctor and the watchdog. Kept out so the Protocol stays a *behavioral*
  contract; a Codex driver will expose the same names anyway.

``AskResult`` is re-exported here (not redefined) because it is the shared
outcome vocabulary — ``ask_health``, ``delivery_guard`` and the cron
statistics are all keyed off ``status``. Any second engine maps its own
outcomes onto THESE statuses, which is what lets the whole health/delivery
layer stay engine-agnostic with zero edits.
"""

# NOTE: deliberately no `from __future__ import annotations` here.
# claude_session.py does not use it either, so keeping annotations as real
# objects lets tests/test_engine.py compare this Protocol's signatures to
# ClaudeSession's with plain equality instead of string-matching guesswork.
from typing import Protocol, runtime_checkable

from d_brain.services.claude_session import DEFAULT_TIMEOUT, AskResult

__all__ = ["AskResult", "EngineDriver"]


@runtime_checkable
class EngineDriver(Protocol):
    """One brain session, whatever process actually runs it.

    ``runtime_checkable`` only gives ``isinstance`` a method-presence check
    (it never inspects signatures) — the real conformance check lives in
    ``tests/test_engine.py``, which compares full signatures against
    ``ClaudeSession``.
    """

    # ── the turn ─────────────────────────────────────────────────────
    def ask(
        self,
        prompt: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        request_id: str | None = None,
        wrap: bool = True,
    ) -> AskResult:
        """Run one turn and return its outcome. Never raises, never blocks
        forever; every failure mode arrives as an ``AskResult`` status."""
        ...

    # ── busy / steering gates ────────────────────────────────────────
    def is_turn_active(self) -> bool:
        """True iff a turn is in flight *as far as the ask-lock knows* — the
        busy-guard the bot handlers use before starting a new turn."""
        ...

    def is_pane_turn_active(self) -> bool:
        """True iff the ENGINE itself is still working, regardless of who
        holds the ask-lock (an unattended cascade holds no lock). The name
        outlives tmux on purpose — for a process-per-turn engine this is
        "the exec process is alive"."""
        ...

    def is_steerable_turn(self) -> bool:
        """True iff the in-flight turn may receive extra user input (a
        maintenance turn must not be contaminated)."""
        ...

    def steer(self, text: str) -> None:
        """Inject text into a live turn, without taking the lock."""
        ...

    def interrupt(self) -> None:
        """Stop the current response (Escape for the TUI, SIGINT for exec)."""
        ...

    def send_control(self, text: str) -> None:
        """A client-side control command (``/clear``, ``/model``) —
        fire-and-forget, produces no model turn. Semantics are
        engine-specific by definition."""
        ...

    # ── health / recovery ────────────────────────────────────────────
    def is_working(self) -> bool:
        """The health-probe view of "a turn is running" — change-aware, used
        by the watchdog to tell a live turn from a wedged one. (This is the
        plan's ``is_turn_health`` slot; the real method is named
        ``is_working``.)"""
        ...

    def force_recover(self) -> bool:
        """Tear down and recreate the session if no live turn holds it.
        False ⇒ a real turn is in flight and nothing was touched."""
        ...

    def ensure_session(self) -> None:
        """Start the session if it is not up. Idempotent."""
        ...

    def is_healthy(self) -> bool:
        """Cheap liveness probe: does the backing session exist at all."""
        ...

    def kill(self) -> None:
        """Tear the session down (CLI/teardown use)."""
        ...

    # ── delivery salvage ─────────────────────────────────────────────
    def capture_text(self) -> str:
        """Raw text of the engine's current output surface (read-only).

        Today's tmux leak point: callers classify this string with
        ``tmux_parse``. Phase 2 keeps that classification on the Claude side
        only — a Codex driver returns its own transcript text and is never
        handed to those classifiers."""
        ...

    def last_reply_for_resend(self) -> tuple[str, str | None]:
        """``(status, body)`` for ``/resend`` — a read-only escape hatch that
        must never wait on the turn lock."""
        ...

    def pop_orphan_replies(self) -> list[str]:
        """Every completed reply nobody's ``ask()`` is waiting on, oldest
        first, consumed exactly once."""
        ...
