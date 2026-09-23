"""Graceful stop: a restart stops taking new work instead of cutting it off.

Step 3 of the reliability plan (``thoughts/projects/agent-infra-backlog.md``
item 33). Step 1 made sure a reply that was BORN could not be lost on the way
out; step 2 made sure a message that ARRIVED could not be lost on the way in.
This one covers the moment between them — the turn that is *in flight* when
the owner restarts the bot, which he does several times a day (six, on
2026-09-22).

What used to happen: ``systemctl restart`` sends SIGTERM, aiogram's own signal
handler stops polling, ``start_polling`` returns, the loop closes and every
handler task dies where it stood. An answer that was thirty seconds from being
finished was simply gone. Nothing was *lost* after step 2 — the message came
back on the next boot and was answered from scratch — but the work, and the
minutes the owner had already waited, were.

The shape, in order
-------------------

1. **First signal** — ``Shutdown.request``. Nothing is killed. A deadline is
   armed (``shutdown_grace_seconds``) and the bot stops *taking* work.
2. **New messages keep arriving and keep being accepted to disk**, because
   polling is deliberately still running. What they do NOT get is a turn:
   ``stop_middleware`` sits outside everything (outside the inbox's own
   middleware, outside auth, outside the routers) and refuses them with one
   short line — "перезапускаюсь" — instead of silence. Refusing there, and
   only there, is what keeps the message: the inbox entry is never claimed and
   never signed off, so ``inbox.replay`` answers it at the next boot.
3. **Turns already running get the whole deadline to finish.** A reply that
   lands in that window goes out through the durable outbox exactly as it
   always did.
4. **Past the deadline** polling stops, the stragglers are cancelled (their
   inbox entries stay behind, unanswered and therefore replayable), the outbox
   gets one last bounded drain, and the process leaves.
5. **A second signal at any point exits immediately.** The owner is in a
   hurry; both queues are on disk; there is nothing to wait for.

Why the exit is ``os._exit``
----------------------------

Returning normally is not good enough, and not because of style. Chat turns
run the engine through ``asyncio.to_thread``; a wedged pane means a worker
thread that cancellation cannot touch. ``asyncio.run`` ends by awaiting
``loop.shutdown_default_executor()``, which on 3.12+ joins those threads with
a 300-second timeout of its own — so a "graceful" return could sit there long
past any ``TimeoutStopSec`` and be SIGKILLed anyway, which is the hang this
step exists to remove. Everything that matters is already on disk by then, so
leaving at once costs nothing and is the only bound that actually holds.

Three things this does NOT cover, stated so they are not mistaken for
oversights (all three surfaced in the blind review):

* **The boot window.** ``install`` is called only after ``inbox.replay`` has
  finished, so a signal during the first minutes of a boot still kills the
  process outright, exactly as before. That is deliberate: ``replay`` signs
  every entry off with a belt-and-braces ``mark_handled`` after feeding it, so
  a refusal from this module DURING a replay would turn "kept the message"
  into "deleted the message" — the one outcome the whole step exists to
  prevent. Closing the boot window properly means teaching ``replay`` the
  difference between new work and work being finished, and belongs with the
  ``msg_id`` idempotency key ``inbox.py`` already names.
* **Ctrl-C in a foreground dev run** now arms the grace like any other signal,
  so a single Ctrl-C looks like a hang and you need two.
  ``DBRAIN_SHUTDOWN_GRACE=0`` restores the old instant stop for development.
* **A straggler that already queued its reply.** A turn cancelled at the
  deadline may have enqueued into the outbox just before the cancel; that
  reply goes out while the inbox entry is released, so the next boot replays
  the message and answers it again. The window is the same one a hard kill
  always had — what is new is that such a turn is now by construction a long
  one. The fix is the same ``msg_id`` key, not more logic here.

Not here on purpose: a shutdown state machine, per-turn deadlines, refusing
work with a nice HTTP-ish status. One module, one middleware, one deadline.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import time
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

# Five minutes. Long enough that the overwhelming majority of turns finish
# inside it (``chat_turn_timeout`` is 1500s, but a normal answer is tens of
# seconds), short enough that a restart with something genuinely wedged is
# still a restart and not an outage. It is only ever PAID when a turn is
# actually running: an idle bot stops the instant it is asked to.
DEFAULT_GRACE = 300.0

# The last outbox pass, after polling has stopped. Bounded, because it talks
# to Telegram and the whole point of this module is that shutting down has a
# known ceiling. Anything it does not manage to send stays queued and goes out
# on the next boot's first drain.
FINAL_DRAIN_TIMEOUT = 10.0

# How long the stragglers get to notice they were cancelled. Cosmetic by
# design — it lets ``handled_middleware``'s CancelledError branch run and log
# that the message is being left unanswered. A turn stuck in a thread will not
# come back in five seconds, or at all, and does not need to.
TURN_CANCEL_TIMEOUT = 5.0

# aiogram's own stop handshake: ``stop_polling`` sets its stop event and then
# waits for ``start_polling``'s finally (cancel the poller, emit shutdown) to
# signal back. It returns in milliseconds in practice — but it is an await on
# another task's completion, i.e. the one place in this sequence that could
# in principle not come back, and the fallback for "did not come back" is now
# systemd's SIGKILL six minutes later rather than forty-five seconds (blind
# review 4). Bounded, with ``poller.cancel()`` right behind it.
POLLING_STOP_TIMEOUT = 10.0

# Everything the stop can cost AFTER the grace itself. The unit's
# TimeoutStopSec must clear ``grace + STOP_OVERHEAD`` or systemd kills the
# process mid-sequence and the wait buys nothing.
STOP_OVERHEAD = POLLING_STOP_TIMEOUT + TURN_CANCEL_TIMEOUT + FINAL_DRAIN_TIMEOUT

# What the shipped units actually say. Kept here, next to the numbers it has
# to clear, so the unit files, the startup warning and the test that pins them
# together all read the same constant instead of three copies of 360 (blind
# review 6).
UNIT_TIMEOUT_STOP_SEC = 360.0

# Exit status. The unit already whitelists it (``SuccessExitStatus=143
# SIGTERM``) because that is what a SIGTERM-terminated process reported before
# this module existed, and a clean stop must not fire the OnFailure alert.
EXIT_CODE = 143

NOTICE = (
    "🔄 Перезапускаюсь. Сообщение принято и сохранено — "
    "отвечу на него сразу после старта."
)


def _event_chat_id(event: Any) -> int | None:
    """Chat behind an aiogram ``Update``, for a human-sent event only.

    Deliberately narrow: this feeds a courtesy notice, and a ``my_chat_member``
    or a service update has nobody sitting there waiting for one.
    """
    message = getattr(event, "message", None)
    if message is None:
        callback = getattr(event, "callback_query", None)
        message = getattr(callback, "message", None)
    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", None)
    return int(chat_id) if isinstance(chat_id, int) else None


class Shutdown:
    """The stop itself: one flag, one deadline, one set of running turns.

    Knows nothing about aiogram or Telegram — the wiring below does — so it is
    testable with a fake clock and without a bot.
    """

    def __init__(
        self,
        *,
        grace: float = DEFAULT_GRACE,
        clock_fn: Callable[[], float] = time.monotonic,
        exit_fn: Callable[[int], None] | None = None,
    ) -> None:
        self.grace = float(grace)
        # Awaited by ``run_bot`` next to the polling task: whichever finishes
        # first decides whether this is a stop or a polling failure.
        self.requested = asyncio.Event()
        self.notified: set[int] = set()
        self._clock = clock_fn
        self._exit = exit_fn if exit_fn is not None else os._exit
        self._deadline: float | None = None
        self._turns: set[asyncio.Task[Any]] = set()

    # ── state ────────────────────────────────────────────────────────

    @property
    def stopping(self) -> bool:
        return self.requested.is_set()

    @property
    def in_flight(self) -> int:
        return len([task for task in self._turns if not task.done()])

    def remaining(self) -> float:
        """Seconds left of the deadline. The full grace before it is armed."""
        if self._deadline is None:
            return self.grace
        return max(0.0, self._deadline - self._clock())

    # ── the two signals ──────────────────────────────────────────────

    def request(self, reason: str = "SIGTERM") -> None:
        """First signal: stop taking work, arm the deadline. Second: leave.

        Both signals land here, and the second one is recognized by the flag
        the first one set, so the caller installs one handler and not two.
        """
        if self.stopping:
            self.force(reason)
            return
        self._deadline = self._clock() + self.grace
        self.requested.set()
        logger.warning(
            "shutdown: %s — not taking new work; %d turn(s) in flight have "
            "%.0fs to finish",
            reason,
            self.in_flight,
            self.grace,
        )

    def force(self, reason: str = "SIGTERM") -> None:
        """Leave now. Nothing is lost: an unfinished turn's message is still
        in the inbox (unanswered, so the next boot replays it) and a reply
        that was already produced is in the outbox."""
        logger.warning(
            "shutdown: %s again — exiting now, %d turn(s) unfinished (their "
            "messages stay in the inbox and are replayed at the next start)",
            reason,
            self.in_flight,
        )
        self._exit(EXIT_CODE)

    def finish(self) -> None:
        """Leave after a completed graceful stop — see the module docstring
        for why this is a hard exit and not a return."""
        logger.info("shutdown: done, exiting")
        self._exit(EXIT_CODE)

    # ── the turns in flight ──────────────────────────────────────────

    def track(self, task: asyncio.Task[Any]) -> None:
        self._turns.add(task)

    def untrack(self, task: asyncio.Task[Any]) -> None:
        self._turns.discard(task)

    async def wait_for_turns(self) -> list[asyncio.Task[Any]]:
        """Block until every tracked turn is done or the deadline runs out.

        Returns whatever is still unfinished, for the caller to cancel. The
        loop re-reads the set each pass because a turn can still be finishing
        (and untracking itself) while another one is waited on.
        """
        while True:
            live = {task for task in self._turns if not task.done()}
            if not live:
                return []
            left = self.remaining()
            if left <= 0:
                logger.warning(
                    "shutdown: %d turn(s) did not finish inside %.0fs — their "
                    "messages stay in the inbox and are replayed at the next "
                    "start",
                    len(live),
                    self.grace,
                )
                return list(live)
            logger.info("shutdown: waiting up to %.0fs for %d turn(s)", left, len(live))
            await asyncio.wait(live, timeout=left)


async def cancel_turns(
    tasks: list[asyncio.Task[Any]], *, timeout: float = TURN_CANCEL_TIMEOUT
) -> None:
    """Cancel the stragglers and give them a moment to unwind."""
    if not tasks:
        return
    for task in tasks:
        task.cancel()
    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout)


# ── wiring ───────────────────────────────────────────────────────────


def install(sd: Shutdown, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Take SIGTERM/SIGINT over from aiogram.

    aiogram's own handler (``start_polling(handle_signals=True)``, the default)
    stops polling immediately — which is exactly the behavior being replaced,
    so the caller passes ``handle_signals=False`` and installs this instead.
    Polling has to keep running through the grace window: that is what lets a
    message arriving mid-restart be accepted to disk and answered with a line
    instead of silence.
    """
    loop = loop or asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):  # pragma: no cover
            loop.add_signal_handler(sig, sd.request, sig.name)


def stop_middleware(
    sd: Shutdown,
    bot: Any,
    *,
    is_allowed: Callable[[Any], bool],
) -> Callable[..., Awaitable[Any]]:
    """Outer middleware for ``dp.update.outer_middleware(...)``.

    Registered before the inbox's, and therefore OUTSIDE it — that is the
    property the design needs, and the only one it may rely on. (It is not the
    outermost middleware in the chain: ``Dispatcher.__init__`` puts its own
    errors/user-context/FSM ones ahead of any user code. None of them
    interferes — blind review 9.) The placement is the whole design:

    * refusing here means the inbox never claims the entry and never signs it
      off, so the message stays on disk and ``inbox.replay`` answers it at the
      next boot. Refusing anywhere inside the inbox middleware would mark it
      handled and delete the very message this step is protecting;
    * accepting here registers the handler task, which is what
      ``wait_for_turns`` waits on.

    ``is_allowed`` is the auth check, passed in rather than re-derived: the
    real auth middleware runs INSIDE this one, so without it a stranger's
    message during a restart would get a polite "перезапускаюсь" it should
    never see.
    """

    async def middleware(handler: Any, event: Any, data: dict[str, Any]) -> Any:
        if sd.stopping:
            await _notify_sender(sd, bot, event, is_allowed=is_allowed)
            return None
        task = asyncio.current_task()
        if task is None:  # pragma: no cover — aiogram always runs in a task
            return await handler(event, data)
        sd.track(task)
        try:
            return await handler(event, data)
        finally:
            sd.untrack(task)

    return middleware


async def _notify_sender(
    sd: Shutdown, bot: Any, event: Any, *, is_allowed: Callable[[Any], bool]
) -> None:
    """One short line per chat per shutdown. Never raises.

    Goes through ``send_response``, i.e. through the durable outbox: if the
    process leaves before Telegram takes it, the notice goes out with the
    next boot's first drain rather than vanishing.
    """
    chat_id = _event_chat_id(event)
    if chat_id is None or chat_id in sd.notified:
        return
    if not is_allowed(event):
        return
    sd.notified.add(chat_id)
    from d_brain.bot.formatters import send_response

    try:
        await send_response(bot, chat_id, NOTICE)
    except Exception:  # noqa: BLE001 — a courtesy line must not break the stop
        logger.warning("shutdown: could not send the restart notice", exc_info=True)


async def stop(
    bot: Any, dp: Any, box: Any, sd: Shutdown, poller: asyncio.Task[Any]
) -> None:
    """The stop sequence itself. Never raises — the caller is exiting anyway.

    Order matters and is the opposite of the obvious one: turns first, polling
    second. Stopping polling first would be simpler and would make the grace
    window silent — the messages arriving during it would sit at Telegram,
    unseen, and nobody would be told anything.

    Every step is bounded. That is not tidiness: the fallback for a step that
    does not come back is now systemd's SIGKILL at ``TimeoutStopSec`` (six
    minutes), where it used to be forty-five seconds.
    """
    from d_brain.services import outbox
    from d_brain.services.systemd_notify import notify

    # A no-op on these units (Type=simple, no NotifyAccess, so NOTIFY_SOCKET
    # is unset) — exactly like the READY=1 and WATCHDOG=1 calls next to it. It
    # is here so that a later move to Type=notify needs no second thought.
    notify("STOPPING=1")
    unfinished = await sd.wait_for_turns()

    with contextlib.suppress(Exception):
        await asyncio.wait_for(dp.stop_polling(), POLLING_STOP_TIMEOUT)
    poller.cancel()
    # CancelledError here is the POLLER's, re-raised into this coroutine by
    # the await — suppressing it is what the cancel above is for, not a
    # swallowed cancellation of the stop itself.
    with contextlib.suppress(Exception, asyncio.CancelledError):
        await poller

    await cancel_turns(unfinished)

    # What the last turns produced leaves now. What does not fit in the window
    # is still on disk and goes out at the next boot.
    with contextlib.suppress(Exception):
        await asyncio.wait_for(outbox.drain(bot, box), FINAL_DRAIN_TIMEOUT)
