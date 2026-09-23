"""Cron runner: the in-bot ticker that fires scheduled jobs in the cron
brain (the second, isolated ClaudeSession).

Semantics:
* jobs.json is re-read every tick (hot-reload — the brain edits it via
  the CLI while the bot runs).
* At-most-once: next_run is advanced and persisted BEFORE ask(), so a
  crash mid-run never refires the same slot.
* Self-healing lives here, not in the watchdog: a failed ask() recovers
  the cron session; max_consecutive_errors auto-disables the job and
  alerts the admin. rate_limited is not the job's fault — skip, wait.
* A reply starting with [SILENT] is not delivered (anti-spam for
  monitoring jobs).
"""

import asyncio
import contextlib
import copy
import html
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from d_brain.services.cron_store import CronJob, CronStore, compute_next_run
from d_brain.services.tmux_parse import PaneState, classify_state, reset_epoch

logger = logging.getLogger(__name__)

SILENT_MARKER = "[SILENT]"

# agent-infra-backlog item 23 (2026-09): the cron pane has no watchdog of its
# own (see runtime.py — a separate tmux pane + runtime_dir from the main
# session's watchdog target). Claude Code never re-checks a rate limit on its
# own — only on new input — so a stale RATE_LIMITED banner left the cron
# brain permanently parked once (2026-08-28, 6-day deadlock). These mirror
# the main watchdog's DEFAULT_LIMIT_MAX_WAIT / DEFAULT_NUDGE_COOLDOWN* —
# fallback wait when the banner's reset time can't be read, and backoff
# between recovery attempts.
LIMIT_MAX_WAIT = 6 * 3600.0
CLEAR_COOLDOWN = 600.0
CLEAR_COOLDOWN_MAX = 3600.0
# Escalating alert (Fix B): if the cron session has been parked at a limit
# banner this long, tell the admin once — then no more than once a day.
LIMIT_ALERT_AFTER = 6 * 3600.0
LIMIT_ALERT_REPEAT = 24 * 3600.0


def wrap_job_prompt(job_id: str, prompt: str, *, scheduled_for: str | None) -> str:
    """Per-run envelope for the cron brain. The marker instruction is NOT
    duplicated here — ask(wrap=True) appends it."""
    return (
        f"[CRON JOB {job_id}] Это плановый запуск по расписанию "
        f"(scheduled_for: {scheduled_for}), не сообщение пользователя.\n\n"
        f"{prompt}\n\n"
        "Правила запуска: ответ уйдёт в Telegram — форматируй в HTML "
        "(<b> <i> <code> <a>), без Markdown. Если доставлять нечего — начни "
        f"ответ строкой {SILENT_MARKER}. В этом запуске ЗАПРЕЩЕНО создавать, "
        "изменять или удалять cron-задания (никаких d_brain.cron add/remove)."
    )


def _find(jobs: list[CronJob], job_id: str) -> CronJob | None:
    return next((j for j in jobs if j.id == job_id), None)


class CronRunner:
    def __init__(
        self,
        store: CronStore,
        session: Any,
        *,
        deliver: Callable[[int, str], Awaitable[None]],
        alert: Callable[[str], Awaitable[None]],
        default_chat_id: int | None,
        job_timeout: float = 600.0,
        max_consecutive_errors: int = 3,
        retry_seconds: float = 300.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.session = session
        self.deliver = deliver
        self.alert = alert
        self.default_chat_id = default_chat_id
        self.job_timeout = job_timeout
        self.max_consecutive_errors = max_consecutive_errors
        self.retry_seconds = retry_seconds
        self.clock = clock or (lambda: datetime.now(UTC))
        self._warned_dormant: set[str] = set()
        # Fix A/B state (agent-infra-backlog item 23): persisted in
        # cron_dir so it survives a bot restart while the pane is still
        # parked at the same banner. In-memory-only cooldown counters below
        # cost at most one extra /clear after a restart, never a correctness
        # issue (same tradeoff watchdog.py makes for its own low-context
        # streak).
        self._limited_since_file = store.cron_dir / "limited_since"
        self._limit_alert_file = store.cron_dir / "limit_alert_ts"
        self._clear_attempts = 0
        self._last_clear_ts = 0.0

    # ── scheduling ───────────────────────────────────────────────────

    def claim_due(self, now: datetime) -> list[CronJob]:
        """Snapshot due jobs and advance their next_run under the lock.

        Persisting the advance BEFORE execution gives at-most-once: a
        crash mid-ask never refires the slot. One-shot ('at') jobs are
        re-armed to now+retry (not None) — so a crash between claim and
        record leaves a job that fires again instead of a bricked one;
        success then deletes it or clears next_run.
        """
        claimed: list[CronJob] = []

        def advance(jobs: list[CronJob]) -> None:
            for job in jobs:
                if not job.enabled or not job.state.next_run:
                    # A recurring job without next_run can never fire —
                    # broken state (hand-edited jobs.json), unlike an
                    # at-job, which legitimately ends with None. Warn
                    # once, not every tick.
                    if (
                        job.enabled
                        and job.schedule.kind != "at"
                        and job.id not in self._warned_dormant
                    ):
                        self._warned_dormant.add(job.id)
                        logger.warning(
                            "job %s is enabled but has no next_run — "
                            "it will never fire; re-add or enable it "
                            "via the cron CLI",
                            job.id,
                        )
                    continue
                try:
                    due_at = datetime.fromisoformat(job.state.next_run)
                except ValueError:
                    # One malformed entry must not stall the whole schedule.
                    logger.error(
                        "job %s has malformed next_run %r — skipping",
                        job.id,
                        job.state.next_run,
                    )
                    continue
                if due_at.tzinfo is None:
                    due_at = due_at.replace(tzinfo=UTC)
                if due_at > now:
                    continue
                claimed.append(copy.deepcopy(job))
                if job.schedule.kind == "at":
                    retry_at = now + timedelta(seconds=self.retry_seconds)
                    job.state.next_run = retry_at.isoformat()
                else:
                    nxt = compute_next_run(job.schedule, now=now)
                    job.state.next_run = nxt.isoformat() if nxt else None

        self.store.mutate(advance)
        return claimed

    # ── execution ────────────────────────────────────────────────────

    async def run_job(self, job: CronJob) -> None:
        wrapped = wrap_job_prompt(
            job.id, job.prompt, scheduled_for=job.state.next_run
        )
        res = await asyncio.to_thread(
            self.session.ask,
            wrapped,
            timeout=self.job_timeout,
            request_id=f"cron-{job.id}",
        )
        now = self.clock()

        if res.status == "rate_limited":
            logger.warning("cron job %s skipped: subscription rate limit", job.id)
            self._note_rate_limited(now)
            self._record_failure(job, now, status="rate_limited", count=False)
            return

        # Any other outcome (ok / error / logged_out / timeout) means the
        # pane is no longer parked at a stale limit banner — a subsequent
        # ask() re-checks the limit on send, so seeing one through here is
        # proof the limit (if it was ever real) is gone. Drop the anchor so
        # the NEXT rate_limited hit starts its own wait, not a leftover one.
        self._clear_limit_state()

        if res.ok:
            reply = (res.reply or "").strip()
            silent = reply.startswith(SILENT_MARKER)
            delivery_error: str | None = None
            if reply and not silent:
                chat = job.chat_id or self.default_chat_id
                if chat is not None:
                    try:
                        await self.deliver(chat, reply)
                    except Exception as exc:  # noqa: BLE001 — Telegram hiccup
                        logger.exception("cron job %s delivery failed", job.id)
                        delivery_error = str(exc)[:200]
                    # NOTE (2026-09-22, durable outbox): `deliver` is
                    # `send_response`, which now queues the reply before it
                    # sends it — a Telegram hiccup no longer raises here, it
                    # becomes a retry. So this branch narrowed to "could not
                    # even be queued". A reply that ends up undeliverable is
                    # NOT the job's fault and deliberately no longer counts
                    # toward max_consecutive_errors; it is reported by the
                    # outbox itself, in /work's dead-queue line.
            # Jobs are stateless by contract; drop the turn's context so
            # the next job starts clean and the window never grows. Best
            # effort — a failed /clear must not block the state update.
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self.session.send_control, "/clear")
            if delivery_error is not None:
                # The work happened but the user never saw it: keep the job
                # (one-shots retry) and count the failure.
                self._record_failure(
                    job, now, status="deliver_error", detail=delivery_error, count=True
                )
            else:
                self._record_success(job, now, silent=silent)
            return

        logger.error("cron job %s failed: %s %s", job.id, res.status, res.detail)
        await asyncio.to_thread(self.session.force_recover)
        disabled = self._record_failure(
            job, now, status=res.status, detail=res.detail, count=True
        )
        if disabled:
            await self.alert(
                f"⛔ Cron job <code>{job.id}</code> отключён после "
                f"{self.max_consecutive_errors} ошибок подряд "
                f"(последняя: {res.status}). Включить: "
                f"<code>python -m d_brain.cron enable {job.id}</code>"
            )

    # ── state updates (single write path: store.mutate) ─────────────

    def _record_success(self, job: CronJob, now: datetime, *, silent: bool) -> None:
        def fn(jobs: list[CronJob]) -> None:
            current = _find(jobs, job.id)
            if current is None:
                return
            if job.delete_after_run:
                jobs.remove(current)
                return
            current.state.last_run = now.isoformat()
            current.state.last_status = "ok-silent" if silent else "ok"
            current.state.last_error = None
            current.state.consecutive_errors = 0
            if current.schedule.kind == "at":
                # Claim re-armed a retry slot; the one-shot has now fired.
                current.state.next_run = None

        self.store.mutate(fn)

    def _record_failure(
        self,
        job: CronJob,
        now: datetime,
        *,
        status: str,
        detail: str | None = None,
        count: bool,
    ) -> bool:
        """Update job state after a non-ok run; True if it got disabled."""
        disabled = False

        def fn(jobs: list[CronJob]) -> None:
            nonlocal disabled
            current = _find(jobs, job.id)
            if current is None:
                return
            current.state.last_run = now.isoformat()
            current.state.last_status = status
            current.state.last_error = detail
            if count:
                current.state.consecutive_errors += 1
                if current.state.consecutive_errors >= self.max_consecutive_errors:
                    current.enabled = False
                    disabled = True
            # A claimed one-shot lost its next_run; re-arm a retry unless
            # the job just got disabled.
            if current.schedule.kind == "at" and not disabled:
                retry_at = now + timedelta(seconds=self.retry_seconds)
                current.state.next_run = retry_at.isoformat()

        self.store.mutate(fn)
        return disabled

    # ── rate-limit self-recovery (Fix A/B, agent-infra-backlog item 23) ──

    def _read_limited_since(self) -> float | None:
        try:
            return float(self._limited_since_file.read_text().strip())
        except (OSError, ValueError):
            return None

    def _write_limited_since(self, value: float) -> None:
        self._limited_since_file.parent.mkdir(parents=True, exist_ok=True)
        self._limited_since_file.write_text(f"{value}\n")

    def _read_limit_alert_ts(self) -> float:
        try:
            return float(self._limit_alert_file.read_text().strip())
        except (OSError, ValueError):
            return 0.0

    def _write_limit_alert_ts(self, value: float) -> None:
        self._limit_alert_file.parent.mkdir(parents=True, exist_ok=True)
        self._limit_alert_file.write_text(f"{value}\n")

    def _note_rate_limited(self, now: datetime) -> None:
        """First rate_limited hit anchors ``limited_since`` — a repeat hit
        must NOT overwrite it, or every retry would push the recovery
        deadline (and the Fix B alert clock) further out forever."""
        if self._read_limited_since() is not None:
            return
        self._write_limited_since(now.timestamp())

    def _clear_limit_state(self) -> None:
        """Called on every non-rate_limited ask() outcome: proof the pane is
        no longer parked at a stale banner, so the anchor (and any pending
        escalation) must not leak into the NEXT limit hit's timers."""
        self._limited_since_file.unlink(missing_ok=True)
        self._limit_alert_file.unlink(missing_ok=True)
        self._clear_attempts = 0

    async def _maybe_limit_alert(self, now: float, since: float) -> None:
        """Fix B: one aggregated heads-up once the cron session has been
        stuck this long, repeated no more than once a day — persisted so
        the "already alerted" state survives a bot restart."""
        if now - since < LIMIT_ALERT_AFTER:
            return
        last = self._read_limit_alert_ts()
        if now - last < LIMIT_ALERT_REPEAT:
            return
        stuck = [
            j.id
            for j in self.store.load()
            if j.enabled and j.state.last_status == "rate_limited"
        ]
        hours = int((now - since) // 3600)
        await self.alert(
            f"⏳ Крон-сессия стоит на лимите подписки уже ~{hours}ч — задания "
            "не выполняются: "
            + ", ".join(f"<code>{i}</code>" for i in stuck)
            + ". Жду сброса и бужу сам; если лимит давно должен был "
            "сброситься — это залипший банер, можно разбудить вручную "
            "(/clear в крон-панель)."
        )
        self._write_limit_alert_ts(now)

    async def _limit_recovery(self) -> None:
        """Fix A: self-healing for the cron pane's own dedicated failure
        mode (agent-infra-backlog item 23). Unlike the main session, no
        watchdog targets the cron runtime_dir at all (see runtime.py) — and
        Claude Code only ever re-checks a rate limit on new input, so once
        ask() sees RATE_LIMITED it refuses to type anything at all
        (pre-send guard). A stale banner therefore parks the pane forever
        with nothing left to wake it — the 2026-08-28 incident (6 days
        parked past a reset that had long since passed).

        Deliberately does NOT mirror watchdog._handle_rate_limited()'s
        nudge("Continue") — for the cron pane, whatever prompt is currently
        parked in the pane at wake time is stale job content (not a live
        in-progress user request), so "Continue" would execute it outside
        of ask() with no way to capture the reply and a risk of duplicate
        execution on the next scheduled claim. /clear re-checks the limit
        (any input does) and drops the stale context; cron jobs are
        stateless by contract, and this runner already sends /clear after
        every successful job (see run_job) — so this reuses the exact same
        safe mechanism, just triggered from a different place.
        """
        since = self._read_limited_since()
        if since is None:
            return
        cap = await asyncio.to_thread(self.session.capture_text)
        if classify_state(cap) != PaneState.RATE_LIMITED:
            return  # banner is gone — a subsequent successful ask() clears the anchor
        now = self.clock().timestamp()
        await self._maybe_limit_alert(now, since)
        due = since + LIMIT_MAX_WAIT
        reset_at = reset_epoch(cap, seen_at=since)
        if reset_at is not None:
            due = min(due, reset_at)
        if now < due:
            return
        cooldown = min(CLEAR_COOLDOWN * 2**self._clear_attempts, CLEAR_COOLDOWN_MAX)
        if self._last_clear_ts and now - self._last_clear_ts < cooldown:
            return
        if self.session.is_turn_active():
            return  # a live ask() holds the lock — don't interfere, try next tick
        await asyncio.to_thread(self.session.send_control, "/clear")
        self._clear_attempts += 1
        self._last_clear_ts = now
        logger.warning(
            "cron pane parked at rate-limit banner past due — sent /clear to "
            "wake it (attempt %d)",
            self._clear_attempts,
        )

    # ── loop ─────────────────────────────────────────────────────────

    async def _deliver_notices(self) -> None:
        """Forward the cron brain's owner notices (e.g. its pane was parked
        off a background task's view — agent-infra-backlog item 28). The
        watchdog only watches the main brain, so the cron loop does this
        for its own session. Best-effort, never breaks the tick."""
        pop = getattr(self.session, "pop_notices", None)
        if pop is None:
            return
        try:
            notices = list(await asyncio.to_thread(pop))
        except Exception:  # noqa: BLE001
            logger.warning("could not read cron session notices", exc_info=True)
            return
        for text in notices:
            try:
                await self.alert(html.escape(text))
            except Exception:  # noqa: BLE001
                logger.error("cron owner notice lost: %s", text, exc_info=True)

    async def tick(self) -> None:
        await self._deliver_notices()
        await self._limit_recovery()
        for job in self.claim_due(self.clock()):
            try:
                await self.run_job(job)
            except Exception as exc:  # noqa: BLE001 — one job must not kill the batch
                logger.exception("cron job %s crashed", job.id)
                with contextlib.suppress(Exception):
                    self._record_failure(
                        job,
                        self.clock(),
                        status="crash",
                        detail=str(exc)[:200],
                        count=True,
                    )

    async def run(self, tick_seconds: float) -> None:
        logger.info("cron runner started (tick %.0fs)", tick_seconds)
        while True:
            try:
                await self.tick()
            except Exception:  # noqa: BLE001 — one bad tick must not kill the loop
                logger.exception("cron tick failed")
            await asyncio.sleep(tick_seconds)


async def run_cron(settings: Any, bot: Any) -> None:
    """Wire the runner to the real store, cron session and Telegram."""
    from d_brain.bot.formatters import send_response
    from d_brain.services import runtime

    store = CronStore(settings.cron_dir)
    session = runtime.get_cron_session(settings)

    async def deliver(chat_id: int, text: str) -> None:
        await send_response(bot, chat_id, text)

    async def alert(text: str) -> None:
        if settings.admin_chat_id is not None:
            await bot.send_message(settings.admin_chat_id, text)

    runner = CronRunner(
        store,
        session,
        deliver=deliver,
        alert=alert,
        default_chat_id=settings.admin_chat_id,
        job_timeout=settings.cron_job_timeout,
        max_consecutive_errors=settings.cron_max_consecutive_errors,
        retry_seconds=settings.cron_retry_seconds,
    )
    await runner.run(settings.cron_tick_seconds)
