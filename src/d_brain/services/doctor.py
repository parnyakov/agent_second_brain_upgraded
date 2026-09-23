"""Daily self-diagnostic for the persistent Claude session.

The "Doctor" the user asked for: once a day it asks the live session a canary
question (the authoritative check that auth + model + plumbing all work — a
silently-expired login is invisible to `claude auth status`), runs a handful
of cheap local checks, and reports a single 🟢/🔴 message to Telegram.

Run by a systemd timer and as the final step of install (install success ==
first green).
"""

import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CANARY_TOKEN = "DBRAIN_OK"
CANARY_TIMEOUT = 120.0


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""


ENGINE_LABELS = {"claude": "Claude Code", "codex": "Codex"}


@dataclass
class DoctorReport:
    ok: bool
    checks: list[CheckResult] = field(default_factory=list)

    def to_telegram(self) -> str:
        # Plain text: the watchdog alerter posts without parse_mode, so HTML
        # tags would reach the owner literally.
        header = "🟢 Осмотр пройден" if self.ok else "🔴 Осмотр: есть проблемы"
        return header + "\n" + "\n".join(self._lines())

    def to_terminal(self) -> str:
        header = "Осмотр пройден" if self.ok else "Осмотр: есть проблемы"
        return header + "\n" + "\n".join(f"  {line}" for line in self._lines())

    def _lines(self) -> list[str]:
        lines = []
        for c in self.checks:
            lines.append(f"{'✅' if c.ok else '❌'} {c.name}: {c.detail}")
            if not c.ok and c.hint:
                lines.append(f"   → {c.hint}")
        return lines


class Doctor:
    def __init__(
        self,
        session: Any,
        *,
        checks: list[Callable[[], CheckResult]] | None = None,
        canary_token: str = CANARY_TOKEN,
        engine: str = "claude",
        canary_attempts: int = 1,
        retry_delay: float = 20.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.session = session
        self._engine = ENGINE_LABELS.get(engine, engine)
        # Install runs the doctor right after (re)starting the services; a
        # first session boot can eat the canary timeout. Logged-out and
        # rate-limited are definitive and never retried.
        self._canary_attempts = max(1, canary_attempts)
        self._retry_delay = retry_delay
        self._sleep = sleep
        self._checks = checks if checks is not None else []
        self._canary_token = canary_token

    def _canary(self) -> CheckResult:
        for attempt in range(self._canary_attempts):
            if attempt:
                self._sleep(self._retry_delay)
            status, result = self._canary_once()
            if result.ok or status in ("logged_out", "rate_limited"):
                break
        return result

    def _canary_once(self) -> tuple[str, CheckResult]:
        res = self.session.ask(
            f"Reply with exactly {self._canary_token} and nothing else.",
            timeout=CANARY_TIMEOUT,
            request_id="maint-doctor",
        )
        if res.status == "logged_out":
            return res.status, CheckResult(
                "canary", False, f"{self._engine}: нет входа в подписку",
                "на сервере выполните dbrain login, затем dbrain doctor",
            )
        if res.status == "rate_limited":
            return res.status, CheckResult(
                "canary", False, "лимит подписки исчерпан",
                "дождитесь сброса лимита подписки и повторите "
                "dbrain doctor",
            )
        if res.ok and self._canary_token in (res.reply or ""):
            return res.status, CheckResult("canary", True, "сессия отвечает")
        return res.status, CheckResult(
            "canary", False, res.detail or res.status,
            "dbrain repair, через минуту dbrain doctor; "
            "если не помогло, dbrain logs 100",
        )

    def run(self) -> DoctorReport:
        checks = [self._canary()]
        for check in self._checks:
            try:
                checks.append(check())
            except Exception as exc:  # noqa: BLE001 — a check must never crash the doctor
                checks.append(
                    CheckResult(getattr(check, "__name__", "check"), False, str(exc))
                )
        return DoctorReport(ok=all(c.ok for c in checks), checks=checks)


# ── built-in local checks (used by main(); injected/faked in tests) ──────


def check_disk(runtime_dir: Path, min_bytes: int = 500_000_000) -> CheckResult:
    free = shutil.disk_usage(runtime_dir).free
    gb = free / 1_000_000_000
    return CheckResult(
        "disk",
        free >= min_bytes,
        f"{gb:.1f} GB свободно",
        "освободите место на диске сервера",
    )


def check_engine_version(
    engine: str = "claude", engine_bin: str | None = None
) -> CheckResult:
    """The CLI of the CONFIGURED engine is installed and runs.

    Checking ``claude`` on a Codex install (or the reverse) would red-alert a
    healthy server every morning."""
    name = "codex" if engine == "codex" else "claude"
    # Manual runs (ssh, cron) often lack ~/.local/bin in PATH — resolve the
    # binary the way the install lays it out instead of false-alarming.
    bin_ = (
        engine_bin or shutil.which(name) or str(Path.home() / ".local" / "bin" / name)
    )
    hint = "установите заново: bash ~/projects/agent-second-brain/bootstrap.sh"
    try:
        out = subprocess.run(
            [bin_, "--version"], capture_output=True, text=True, timeout=15
        )
        detail = out.stdout.strip() or "ok"
        return CheckResult(name, out.returncode == 0, detail, hint)
    except (OSError, subprocess.SubprocessError) as exc:
        detail = f"не найден ({exc.__class__.__name__})"
        return CheckResult(name, False, detail, hint)


def check_claude_version(claude_bin: str | None = None) -> CheckResult:
    return check_engine_version("claude", claude_bin)


def check_env(settings: Any) -> CheckResult:
    missing = [
        k
        for k, v in {
            "TELEGRAM_BOT_TOKEN": settings.telegram_bot_token,
            "DEEPGRAM_API_KEY": settings.deepgram_api_key,
        }.items()
        if not v
    ]
    return CheckResult(
        "env",
        not missing,
        "все ключи на месте" if not missing else f"нет: {missing}",
        "впишите ключ в ~/projects/agent-second-brain/.env и выполните dbrain restart",
    )


# R7 (Fable audit): weekly, not daily — the doctor's OWN timer is daily
# (dbrain-doctor.timer, 08:00), so this check self-gates via a small state
# file rather than needing new cron/timer infrastructure. A dedicated timer
# was considered and rejected: this piggybacks cleanly on infra that already
# exists, watches, and alerts (per-README "invocable from the existing
# weekly/periodic infra-check flow" — see the change contract for why this
# was judged the cleaner fit over writing into the live brain's own
# cron/jobs.json, which is runtime state outside version control).
DEFAULT_MARKER_COMPLIANCE_INTERVAL = 7 * 24 * 3600.0


def check_marker_compliance(
    session: Any,
    runtime_dir: Path,
    *,
    repo_dir: Path,
    min_interval_seconds: float = DEFAULT_MARKER_COMPLIANCE_INTERVAL,
    clock_fn: Callable[[], float] = time.time,
) -> CheckResult:
    """R7: run `scripts/marker_compliance.py` against the pinned session's
    transcript (R1) at most once per ``min_interval_seconds`` (default:
    weekly), so drop-rate trending is monitored going forward without
    re-parsing a 100MB+ transcript on every daily doctor run.

    Always reports ok=True: this is a TREND metric to watch, not a pass/fail
    gate the audit asked for — a rising drop rate is the owner's call (see
    R3), not a reason to red-alert the whole daily doctor over it.
    """
    state_file = runtime_dir / "last_marker_compliance_check"
    now = clock_fn()
    try:
        last = float(state_file.read_text().strip())
    except (OSError, ValueError):
        last = 0.0
    if now - last < min_interval_seconds:
        return CheckResult("marker_compliance", True, "не пора — раз в неделю")

    try:
        transcript = session.current_transcript_path()
    except Exception as exc:  # noqa: BLE001 — a check must never crash the doctor
        return CheckResult("marker_compliance", True, f"пропущено: {exc}")
    if transcript is None or not Path(transcript).exists():
        return CheckResult("marker_compliance", True, "пропущено: нет транскрипта")

    script = Path(repo_dir) / "scripts" / "marker_compliance.py"
    try:
        out = subprocess.run(
            [sys.executable, str(script), "--transcript", str(transcript)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult("marker_compliance", True, f"не запустился: {exc}")

    try:
        state_file.write_text(f"{now}\n")
    except OSError as exc:
        logger.warning("could not persist last_marker_compliance_check: %s", exc)

    lines = out.stdout.strip().splitlines()
    detail = next(
        (ln for ln in lines if ln.startswith("total marker-wrap turns")),
        lines[-1] if lines else "нет данных",
    )
    return CheckResult("marker_compliance", True, detail)


def run_cli(
    session: Any,
    *,
    checks: list,
    alert: Any,
    engine: str = "claude",
    out: Any = None,
    canary_attempts: int = 1,
) -> int:
    """Run the checks, deliver the report, map health to an exit code —
    upgrade.sh and the systemd OnFailure= hook key off that code. The same
    report is printed, so `dbrain doctor` in a terminal shows every check."""
    report = Doctor(
        session, checks=checks, engine=engine, canary_attempts=canary_attempts
    ).run()
    print(report.to_terminal(), file=out or sys.stdout, flush=True)
    alert(report.to_telegram())
    logger.info("doctor: ok=%s", report.ok)
    return 0 if report.ok else 1


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    from d_brain import logsafe

    logsafe.install()
    from d_brain.config import get_settings
    from d_brain.services.runtime import get_session
    from d_brain.services.watchdog import _telegram_alerter

    settings = get_settings()
    session = get_session(settings)
    checks = [
        lambda: check_disk(settings.runtime_dir),
        lambda: check_engine_version(settings.chat_engine),
        lambda: check_env(settings),
        lambda: check_marker_compliance(
            session, settings.runtime_dir, repo_dir=settings.project_root
        ),
    ]
    raise SystemExit(
        run_cli(
            session, checks=checks, alert=_telegram_alerter(settings),
            engine=settings.chat_engine,
            canary_attempts=int(os.environ.get("DBRAIN_DOCTOR_CANARY_ATTEMPTS", "1")),
        )
    )


if __name__ == "__main__":
    main()
