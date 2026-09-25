"""Canary check for a self-deploy of delivery-critical code.

The underlying problem in one sentence: the session deployed a change to its
own reply path, had no way to test that path except by using it, and so the
first evidence that anything was wrong was a human noticing silence hours
later. The way out is not "test more before deploying" — the pre-deploy test
suite was green and stayed green through the whole outage. It is to make the
deploy itself unfinished until a real round trip has succeeded.

Four things have to hold for a reply to reach the owner, and each one broke
at some point during a real production outage:

  unit_active     — the service is up (and, after a deploy, up SINCE it)
  single_instance — exactly one `python -m d_brain`; a second one is the
                    orphan-under-`uv` bug that survived a restart
  telegram_api    — the token still authenticates and api.telegram.org answers
  brain_canary    — a trivial prompt goes through the live tmux pane, through
                    marker extraction, and comes back — the actual code a
                    delivery-critical deploy just changed
  outbound        — sendMessage to the admin chat returns 200

Checks run in order and stop at the first failure, because everything after a
dead unit is noise.

The canary's own timeout is deliberately SHORT (CANARY_TIMEOUT, 2 min) and has
nothing to do with `DEFAULT_TIMEOUT` (3600s). That ceiling exists for real
tool-dense work; a canary is a no-op turn, so anything slower than a couple of
minutes IS the failure we are looking for. Reusing the runaway ceiling here
would mean waiting an hour to learn the channel is dead — which is roughly how
long it took during the real outage that motivated this canary.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Change any of these and the reply path may stop working in a way no unit
# test can see, because the thing under test is the live pane and the live
# process tree. Verify before calling the deploy done.
DELIVERY_CRITICAL_PATHS = (
    "src/d_brain/services/claude_session.py",
    "src/d_brain/services/chat_session.py",
    "src/d_brain/services/tmux_parse.py",
    "src/d_brain/services/transcript.py",
    "src/d_brain/services/watchdog.py",
    "src/d_brain/services/delivery_guard.py",
    "src/d_brain/services/runtime.py",
    "src/d_brain/services/long_run.py",
    # The durable-delivery modules. A reply now lives in
    # the outbox, an incoming message in the inbox, a message waiting for a
    # busy session in the chat queue, and the stop decides which of them
    # survives a restart — a bug in any of them is a lost message that no
    # unit test sees, which is this list's whole criterion.
    "src/d_brain/services/outbox.py",
    "src/d_brain/services/inbox.py",
    "src/d_brain/services/chat_queue.py",
    "src/d_brain/services/shutdown.py",
    # And the module that reads those queues back as evidence:
    # every alerting path now asks it whether a message was really lost, so
    # a bug here is either a lost message nobody is told about or a false
    # alarm every five minutes.
    "src/d_brain/services/delivery_proof.py",
    "src/d_brain/bot/handlers/chat.py",
    "src/d_brain/bot/formatters.py",
    "src/d_brain/bot/main.py",
    "deploy/dbrain-bot.service",
    # The unit that is actually live on this server (the --user one above is
    # the legacy fallback); its TimeoutStopSec is what bounds the stop.
    "deploy/systemd/dbrain-bot@.service",
    "deploy/dbrain-watchdog.service",
    "deploy/brain-system.md",
)

CANARY_TIMEOUT = 120.0
# How long to wait for a live turn to finish before giving up. A canary must
# never steal the pane from a real conversation.
CANARY_IDLE_WAIT = 180.0
# maint- prefix: marks the turn as maintenance, so a user message arriving
# mid-canary is answered with "фоновое обслуживание" instead of being injected
# into the canary turn (see claude_session.MAINT_PREFIX).
CANARY_RID_PREFIX = "maint-canary-"
CANARY_WORD = "PONG"
CANARY_PROMPT = (
    f"Canary check after a deploy. Reply with exactly the word {CANARY_WORD} "
    "and nothing else. Do not use any tools and do not write any files."
)


def touches_delivery_path(changed: Iterable[str]) -> list[str]:
    """Which of the changed files are on the delivery path.

    Accepts repo-relative paths as `git diff --name-only` prints them.
    """
    critical = set(DELIVERY_CRITICAL_PATHS)
    return sorted({p for p in changed if p.strip() in critical})


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool | None  # None ⇒ not run
    detail: str = ""

    @property
    def mark(self) -> str:
        return {True: "PASS", False: "FAIL", None: "SKIP"}[self.ok]


@dataclass(frozen=True)
class Report:
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks) and bool(self.checks)

    @property
    def failure(self) -> Check | None:
        return next((c for c in self.checks if c.ok is False), None)

    def summary(self) -> str:
        lines = [f"  [{c.mark}] {c.name}: {c.detail}" for c in self.checks]
        head = "deploy-verify: HEALTHY" if self.ok else "deploy-verify: NOT HEALTHY"
        return "\n".join([head, *lines])


Probe = Callable[[], tuple[bool, str]]


def verify(probes: Sequence[tuple[str, Probe]]) -> Report:
    """Run probes in order, stopping at the first failure."""
    checks: list[Check] = []
    stopped = False
    for name, probe in probes:
        if stopped:
            checks.append(Check(name, None, "not run — an earlier check failed"))
            continue
        try:
            ok, detail = probe()
        except Exception as exc:  # noqa: BLE001 — a probe crash IS a failure
            ok, detail = False, f"probe raised: {exc!r}"
        checks.append(Check(name, ok, detail))
        stopped = not ok
    return Report(tuple(checks))


# ── probe factories ──────────────────────────────────────────────────
# Each returns a zero-arg Probe with its I/O injected, so the policy above
# stays testable without a systemd, a network or a tmux pane.


def unit_active_probe(
    show_fn: Callable[[str], dict[str, str]],
    unit: str,
    *,
    restarted_after_monotonic: float | None = None,
) -> Probe:
    """`systemctl --user show <unit>` says active — and, when a reference
    point is given, entered that state after it.

    The reference is CLOCK_MONOTONIC seconds since boot
    (`time.clock_gettime(time.CLOCK_MONOTONIC)`), the same base systemd
    reports in `ActiveEnterTimestampMonotonic` (microseconds). Sampling it
    before the deploy is what turns "the service is up" into "the service is
    running the code I just deployed" — an editor writing new bytes to disk
    proves nothing until the process has been replaced.
    """

    def probe() -> tuple[bool, str]:
        props = show_fn(unit)
        state = props.get("ActiveState", "?")
        sub = props.get("SubState", "?")
        if state != "active":
            return False, f"{unit} is {state} ({sub})"
        if restarted_after_monotonic is not None:
            try:
                entered = float(props.get("ActiveEnterTimestampMonotonic", 0)) / 1e6
            except ValueError:
                entered = 0.0
            if entered < restarted_after_monotonic:
                return (
                    False,
                    f"{unit} is active but has NOT restarted since the deploy — "
                    "it is still running the old code",
                )
        return True, f"{unit} active ({sub})"

    return probe


def single_instance_probe(pids_fn: Callable[[], Sequence[int]]) -> Probe:
    """Exactly one live bot process. Two means the orphaned-child bug: the
    old instance survived a stop and a new one started on top of it."""

    def probe() -> tuple[bool, str]:
        pids = list(pids_fn())
        if len(pids) == 1:
            return True, f"one bot process (pid {pids[0]})"
        if not pids:
            return False, "no `python -m d_brain` process at all"
        return False, f"{len(pids)} bot processes alive — orphan: {pids}"

    return probe


def telegram_api_probe(get_me_fn: Callable[[], tuple[bool, str]]) -> Probe:
    """Token authenticates and api.telegram.org answers. getMe, never
    getUpdates: a second getUpdates caller would steal the running bot's
    updates and trigger 409 conflicts — the check would cause the outage."""

    def probe() -> tuple[bool, str]:
        ok, detail = get_me_fn()
        return ok, detail

    return probe


def brain_canary_probe(
    session,
    *,
    timeout: float = CANARY_TIMEOUT,
    idle_wait: float = CANARY_IDLE_WAIT,
    clock_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    rid_fn: Callable[[], str] = lambda: uuid.uuid4().hex[:8],
) -> Probe:
    """A real round trip through the live pane: prompt in, marker out."""

    def probe() -> tuple[bool, str]:
        deadline = clock_fn() + idle_wait
        while session.is_turn_active():
            if clock_fn() >= deadline:
                return False, (
                    f"a turn held the pane for the whole {idle_wait:.0f}s wait — "
                    "cannot attest the reply path"
                )
            sleep_fn(2.0)
        started = clock_fn()
        res = session.ask(
            CANARY_PROMPT,
            timeout=timeout,
            request_id=f"{CANARY_RID_PREFIX}{rid_fn()}",
        )
        took = clock_fn() - started
        if not res.ok:
            return False, f"canary turn returned {res.status} after {took:.0f}s"
        reply = (res.reply or "").strip()
        if CANARY_WORD not in reply.upper():
            return False, f"canary replied {reply[:80]!r}, expected {CANARY_WORD}"
        return True, f"round trip in {took:.0f}s"

    return probe


def outbound_probe(send_fn: Callable[[str], tuple[bool, str]], text: str) -> Probe:
    """The last hop: a message actually leaves for the admin chat."""

    def probe() -> tuple[bool, str]:
        return send_fn(text)

    return probe
