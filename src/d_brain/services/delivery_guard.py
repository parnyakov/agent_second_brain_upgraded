"""Automatic recovery when the delivery path keeps failing.

Background (after a real production outage):
the bot kept polling Telegram and kept "handling" updates, systemd kept
reporting `active (running)`, and the user still got nothing back for hours.
Every existing health signal was a liveness signal, and liveness was fine.
The only thing that was broken was the outcome of the work.

So this guard watches outcomes, not liveness. It reads the ledger the bot
writes (`ask_health`) and, on N consecutive delivered-nothing turns inside a
short window, restarts `dbrain-bot.service` — the single action that actually
ended that outage, and the same action a human would take first.

Deliberately NOT automated: rolling code back. A restart is idempotent,
bounded, loses at most one in-flight turn, and needs no judgement about which
commit is at fault. `git revert` unattended is none of those things — it
rewrites the repository from a process that is, by construction, already
running in a degraded state and cannot verify its own conclusion. When
restarts stop helping, this escalates to a human with the exact command to
run instead of running it. Autonomy is cheap where the action is reversible;
here it is not.

Restart budget: at most `max_restarts` per streak, `restart_cooldown` apart.
Past that it alerts once and goes quiet, because a restart loop against a
fault a restart cannot fix is worse than the fault.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from d_brain.services import ask_health
from d_brain.services.ask_health import Health
from d_brain.services.claude_session import DEFAULT_STALL_TIMEOUT, DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)

DEFAULT_UNIT = "dbrain-bot.service"
# "user" = today's `systemctl --user`. "system" = templated system units,
# restarted via the sudoers whitelist (plan §3.5) since polkit 0.105 cannot
# grant per-unit rights.
DEFAULT_SCOPE = "user"
# Three in a row. One timeout is a long turn; two can still be one wedged
# task the user retried. Three inside the window is a pattern.
DEFAULT_FAIL_THRESHOLD = 3
# The failures have to be CLOSE together — but "close" is relative to how
# long a single turn is allowed to take before it even COUNTS as a failure.
# Derived, not a bare literal: three genuine failures, each taking up to
# DEFAULT_TIMEOUT (a turn stalls first at DEFAULT_STALL_TIMEOUT, but the hard
# ceiling a slow-but-alive turn can still run to is DEFAULT_TIMEOUT), can
# span up to 3x that before the guard would dismiss them as "too slow to
# alarm". A bare 1800s window here silently stopped matching DEFAULT_STALL_
# TIMEOUT once it was raised 2026-08-20 (180s→900s; DEFAULT_TIMEOUT was
# already 3600s before that change, not raised alongside it): three real
# stalls at the new stall timeout already span ≥1800s, and three real
# timeouts span ~7200s — either way the restart guard would never fire.
# WHOEVER RAISES DEFAULT_STALL_TIMEOUT OR DEFAULT_TIMEOUT AGAIN: this moves
# with them automatically, but re-check DEFAULT_RESTART_COOLDOWN /
# DEFAULT_MAX_RESTARTS still make sense against the new window.
DEFAULT_WINDOW = 3 * max(DEFAULT_STALL_TIMEOUT, DEFAULT_TIMEOUT)
DEFAULT_RESTART_COOLDOWN = 900.0  # 15 min between restarts of one streak
DEFAULT_MAX_RESTARTS = 2

# C4: the unit name and the systemctl flavour used to be baked into these
# strings. They are now filled from the guard's own `_unit`/`_scope`, so a
# templated instance tells the operator to inspect ITS unit, not the
# owner's default one.
# With the defaults (dbrain-bot.service / "user") the rendered text is
# byte-for-byte what it was before this change.
RESTART_MSG = (
    "♻️ Доставка ответов не работала {n} раза подряд — перезапустил "
    "{unit}. Если следующий ответ не придёт, канал сломан глубже."
)
ESCALATE_MSG = (
    "🔴 Доставка не восстановилась после {n} перезапусков подряд "
    "({streak} неудачных ответов). Автоматика дальше не лезет.\n"
    "Проверить: journalctl {journal_flag}-u {unit} -n 100\n"
    "Откатить последний деплой, если он виноват:\n"
    "  git -C {repo} log --oneline -5\n"
    "  git -C {repo} revert --no-edit <sha> && {restart_cmd}"
)


def _journal_flag(scope: str) -> str:
    """`journalctl --user -u X` for user units; plain `-u X` for system ones.
    Trailing space is part of the token so the system form has no double
    space where the flag used to be."""
    return "--user " if scope == "user" else ""


def _restart_cmd(unit: str, scope: str) -> str:
    """The copy-pasteable manual restart for this instance's scope.

    Deliberately WITHOUT `-n`: this string goes into a Telegram message for a
    human to paste into a terminal, where being prompted for a password is
    the correct behavior. `-n` belongs to the unattended path
    (`restart_argv`), where a prompt would hang the watchdog tick.
    """
    if scope == "user":
        return f"systemctl --user restart {unit}"
    return f"sudo systemctl restart {unit}"


@dataclass(frozen=True)
class GuardDecision:
    """What the guard did on one tick, and why."""

    action: str  # "none" | "restart" | "escalate"
    reason: str = ""


class DeliveryGuard:
    """Consecutive-failure guard over the ask-health ledger."""

    def __init__(
        self,
        runtime_dir: Path | str,
        *,
        restart_fn: Callable[[str], bool],
        alert_fn: Callable[[str], None] = lambda _m: None,
        clock_fn: Callable[[], float] = time.time,
        unit: str = DEFAULT_UNIT,
        scope: str = DEFAULT_SCOPE,
        repo_dir: Path | str = "~/projects/dbrain",
        fail_threshold: int = DEFAULT_FAIL_THRESHOLD,
        window: float = DEFAULT_WINDOW,
        restart_cooldown: float = DEFAULT_RESTART_COOLDOWN,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self._restart_fn = restart_fn
        self._alert_fn = alert_fn
        self._clock = clock_fn
        self._unit = unit
        self._scope = scope
        self._repo_dir = str(repo_dir)
        self._threshold = fail_threshold
        self._window = window
        self._cooldown = restart_cooldown
        self._max_restarts = max_restarts
        # Per-streak bookkeeping. Identified by when the streak began, so a
        # NEW streak after a recovery gets a fresh budget without any need to
        # persist state across watchdog restarts.
        self._streak_id: float | None = None
        self._restarts = 0
        self._last_restart_ts = 0.0
        self._escalated = False

    def _reset(self) -> None:
        self._streak_id = None
        self._restarts = 0
        self._last_restart_ts = 0.0
        self._escalated = False

    def decide(self, health: Health) -> GuardDecision:
        """Pure decision over the ledger + this guard's own restart budget.

        THE LEDGER IS THE PROOF, and deliberately so (blind review). It is
        tempting to add a second gate here —
        "an outbox receipt dated after the last failed turn means the
        channel worked, so do not act" — and it is wrong: the apology the
        user gets for a failed turn goes out through the outbox too, and its
        receipt is minted AFTER the ledger row for that same turn. Such a
        gate is therefore satisfied by every streak this guard exists for,
        which disarms it completely while looking like a careful check.

        What was actually wrong, in the incident that motivated this fix,
        was the ledger, and it is fixed where it is written: a turn the
        duty session covered now
        scores `ok`, and so does a late reply the watchdog delivers out of
        the orphan path. `fail_streak >= threshold` now means what it says —
        that many turns in a row where the person got nothing but a
        brush-off, by any delivery route.
        """
        if health.fail_streak < self._threshold:
            # Healthy again (or not yet alarming) — re-arm for next time.
            if health.fail_streak == 0:
                self._reset()
            return GuardDecision("none")

        if health.streak_span > self._window:
            return GuardDecision(
                "none", f"streak spans {health.streak_span:.0f}s — too slow to alarm"
            )

        now = self._clock()
        # A streak that stopped moving is history, not an ongoing outage. It
        # is what a watchdog restarted hours later reads on its first tick,
        # and restarting the bot over yesterday's evidence helps nobody.
        # Nothing is lost by going quiet: delivery-watch.sh reports a standing
        # fail_streak out-of-band, on its own timer.
        age = now - health.last_ts
        if age > self._window:
            return GuardDecision(
                "none", f"last failed turn was {age:.0f}s ago — stale evidence"
            )

        if self._streak_id != health.streak_started_ts:
            # First tick of a new streak that crossed the threshold.
            self._streak_id = health.streak_started_ts
            self._restarts = 0
            self._last_restart_ts = 0.0
            self._escalated = False

        # Never spend another restart — or escalate — without a turn having
        # actually been attempted and failed SINCE the last restart. The
        # ledger only moves when a turn ends, so after a restart with nobody
        # writing in (night, or the owner gave up and went to do something
        # else) it holds exactly the same bytes that triggered the restart.
        # Judging a restart by re-reading its own trigger is how the guard
        # would come to announce "не восстановилась после 2 перезапусков"
        # having never once seen the channel tried.
        if self._restarts and health.last_ts <= self._last_restart_ts:
            return GuardDecision("none", "no failed turn since the last restart")

        if self._restarts >= self._max_restarts:
            if self._escalated:
                return GuardDecision("none", "already escalated for this streak")
            self._escalated = True
            return GuardDecision("escalate", "restarts exhausted")

        if self._restarts and now - self._last_restart_ts < self._cooldown:
            return GuardDecision("none", "waiting out the restart cooldown")

        self._restarts += 1
        self._last_restart_ts = now
        return GuardDecision("restart", f"{health.fail_streak} failed turns in a row")

    def tick(self) -> GuardDecision:
        """Read the ledger and act. Never raises — the watchdog loop that
        calls this must survive anything that happens in here."""
        try:
            health = ask_health.read(self.runtime_dir)
            decision = self.decide(health)
            if decision.action == "restart":
                logger.error(
                    "delivery guard: restarting %s (%s)", self._unit, decision.reason
                )
                if self._restart_fn(self._unit):
                    self._alert_fn(
                        RESTART_MSG.format(n=health.fail_streak, unit=self._unit)
                    )
                else:
                    logger.error("delivery guard: restart of %s failed", self._unit)
            elif decision.action == "escalate":
                logger.error("delivery guard: escalating (%s)", decision.reason)
                self._alert_fn(
                    ESCALATE_MSG.format(
                        n=self._restarts,
                        streak=health.fail_streak,
                        repo=self._repo_dir,
                        unit=self._unit,
                        journal_flag=_journal_flag(self._scope),
                        restart_cmd=_restart_cmd(self._unit, self._scope),
                    )
                )
            return decision
        except Exception:  # noqa: BLE001 — a guard must not kill its host loop
            logger.exception("delivery guard tick failed")
            return GuardDecision("none", "tick raised")


def restart_argv(unit: str, scope: str = DEFAULT_SCOPE) -> list[str]:
    """The exact argv `systemctl_restarter` will run. Split out so both
    branches are unit-testable without spawning systemctl.

    scope="user"   → today's exact command, unchanged.
    scope="system" → `sudo -n systemctl restart --no-block <unit>`. `-n` never
    prompts: a service account has no tty, so a missing sudoers rule must fail
    fast rather than hang the watchdog tick.

    `--no-block` in both: a unit that is slow to stop must not wedge the
    watchdog tick the way a slow stop wedged the bot on 2026-08-20.
    """
    if scope == "user":
        return ["systemctl", "--user", "restart", "--no-block", unit]
    return ["sudo", "-n", "systemctl", "restart", "--no-block", unit]


def systemctl_restarter(
    scope: str = DEFAULT_SCOPE,
) -> Callable[[str], bool]:  # pragma: no cover - I/O
    """Real restart action for the given systemd scope."""
    import subprocess

    def restart(unit: str) -> bool:
        try:
            return (
                subprocess.run(
                    restart_argv(unit, scope),
                    capture_output=True,
                    timeout=30,
                    check=False,
                ).returncode
                == 0
            )
        except Exception:  # noqa: BLE001
            logger.exception("systemctl restart %s failed", unit)
            return False

    return restart
