"""Application configuration using Pydantic Settings."""

import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Defaults to ".env" (relative to cwd) -- today's exact behavior for
    # the owner's own instance, which never sets DBRAIN_ENV_FILE. Multi-instance
    # units share one WorkingDirectory (the shared checkout), so a hardcoded
    # relative ".env" would always resolve to whoever's real .env happens to
    # live there -- unreadable (and wrong) for any other instance. systemd's
    # own EnvironmentFile=/etc/dbrain/%i/.env already lands DBRAIN_ENV_FILE in
    # the process environment before this class is ever instantiated, so
    # reading it here just needs to happen before pydantic-settings tries to
    # open the dotenv file, not duplicate any parsing logic.
    model_config = SettingsConfigDict(
        env_file=os.environ.get("DBRAIN_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_bot_token: str = Field(description="Telegram Bot API token")
    deepgram_api_key: str = Field(description="Deepgram API key for transcription")
    vault_path: Path = Field(
        default=Path("./vault"),
        description="Path to Obsidian vault directory",
    )
    allowed_user_ids: list[int] = Field(
        default_factory=list,
        description="List of Telegram user IDs allowed to use the bot",
    )
    allow_all_users: bool = Field(
        default=False,
        description="Whether to allow access to all users (security risk!)",
    )

    # ── persistent tmux session ──────────────────────────────────────
    runtime_dir: Path = Field(
        default_factory=lambda: Path.home() / ".dbrain",
        description="Runtime dir for locks, pane.log, ready/inflight flags",
    )
    brain_session_name: str = Field(
        default="",
        description="tmux session name (empty → generated & persisted per install)",
    )
    claude_model: str = Field(
        default="",
        description="Model for the session (empty = Claude Code default)",
    )
    tz: str = Field(default="UTC", description="Timezone for timers/reports")

    # ── pane geometry ────────────────────────────────────────────────
    # Was hardcoded (200x50) in claude_session.py. Made settable
    # 2026-09-20 after the second instance stopped delivering replies:
    # at a wide pane the Claude Code TUI can lay the screen out in TWO
    # columns (transcript left, a file diff right), which puts text
    # AFTER the `<<<E:id>>>` marker on its line — and the parser
    # deliberately only accepts a marker at END of line, so every reply
    # parsed as region=None. tmux_parse now strips such a right column
    # (see _strip_right_column), but a narrower pane is the cheap,
    # immediate mitigation that needs no code path to be correct.
    # Defaults are exactly the previous hardcoded values, so leaving
    # both unset is a byte-for-byte no-op.
    pane_width: int = Field(
        default=200,
        ge=40,
        validation_alias=AliasChoices("DBRAIN_PANE_WIDTH", "pane_width"),
        description=(
            "tmux pane width for the interactive session. 200 = today's "
            "hardcoded value. Lower it (e.g. 160) on an instance where "
            "the TUI splits the screen into two columns."
        ),
    )
    pane_height: int = Field(
        default=50,
        ge=20,
        validation_alias=AliasChoices("DBRAIN_PANE_HEIGHT", "pane_height"),
        description=(
            "tmux pane height for the interactive session. 50 = today's "
            "hardcoded value; the TUI draws its footer just below the "
            "content, so a much TALLER pane puts the footer mid-screen "
            "and chrome-region state detection misses it (see "
            "claude_session._PANE_HEIGHT). Do not raise without "
            "re-checking _CHROME_LINES."
        ),
    )

    # ── multi-instance parameterization (C1, second-instance rollout plan) ──
    # Every field here defaults to EXACTLY today's single-instance behavior.
    # Setting none of them must be a byte-for-byte no-op.
    project_root: Path | None = Field(
        default=None,
        description=(
            "Code checkout root — where deploy/brain-system.md, deploy/"
            "tmux.conf and mcp-config.json live. Empty → vault_path.parent, "
            "which is exactly what the code derived before this field "
            "existed. A second instance whose vault lives OUTSIDE the "
            "checkout (e.g. /var/lib/dbrain-<instance>/vault) sets "
            "PROJECT_ROOT to the shared checkout instead."
        ),
    )
    bot_unit: str = Field(
        default="dbrain-bot.service",
        description=(
            "systemd unit the DeliveryGuard restarts and that operator "
            "messages name. Default is today's hardcoded unit name; a "
            "templated instance sets e.g. dbrain-bot@<instance>.service."
        ),
    )
    systemd_scope: Literal["user", "system"] = Field(
        default="user",
        description=(
            "'user' → `systemctl --user restart <unit>` (today's behavior). "
            "'system' → `sudo -n systemctl restart <unit>`, for the "
            "templated system units a non-login service account needs."
        ),
    )
    context_alert_tokens: int = Field(
        default=400_000,
        description=(
            "R3 (Fable audit): one-time watchdog notification threshold for "
            "the estimated context size (cache_read_input_tokens + "
            "cache_creation_input_tokens + input_tokens, B2 fix 2026-08-22 — "
            "cache_read alone undercounts on cache-rewrite turns) on the "
            "latest transcript record — the measured marker-drop rate jumps "
            "~10x above this (1.3% below vs ~12% above, per a prior "
            "production incident)."
        ),
    )
    long_run_alert_seconds: float = Field(
        default=900.0,
        description=(
            "How long an UNATTENDED "
            "long turn (pane's main turn active, ask-lock free — the shape "
            "of a multi-level autonomous agent cascade) runs before the "
            "watchdog sends one heads-up notice and writes the long-run.json "
            "marker claude_session.ask() uses to short-circuit its own "
            "busy-wait. 0 disables long-run tracking, the fast busy-active "
            "path, and the notices entirely — the rollback switch for this "
            "feature. The progress-aware busy/busy_active classification "
            "fix itself has no flag and stays active regardless."
        ),
    )
    long_run_nudge_seconds: float = Field(
        default=0.0,
        description=(
            "Step E (optional, default OFF): when > 0 and an unattended "
            "turn passes this (larger than long_run_alert_seconds) "
            "threshold, the watchdog steers a one-shot reminder into the "
            "pane telling the session to close the turn and dispatch "
            "remaining work to background agents. 0 (the default) means "
            "this code path never fires — zero live behavior change."
        ),
    )
    long_run_max_seconds: float = Field(
        default=1800.0,
        description=(
            "HARD cap on how long an "
            "UNATTENDED turn of the main session may run before the watchdog "
            "closes it automatically (one interrupt() per run) and tells the "
            "owner why. The owner's rule: a long background job belongs to "
            "an agent, not to the main session — a turn that outlives this "
            "cap is a failure mode, not work. MUST be larger than "
            "long_run_alert_seconds (and larger than long_run_nudge_seconds "
            "when that is enabled): the intended order of the three "
            "thresholds is alert → nudge → max. 0 disables the auto-close "
            "entirely — the rollback switch for this feature; the heads-up "
            "alert and the marker keep working regardless."
        ),
    )

    # ── engine selection (Codex-engine plan, phase 1) ────────────────
    # PER-SESSION on purpose, not one global switch: the pilot runs Codex on
    # the isolated cron session while the owner's chat brain stays on Claude Code
    # the whole time. Both default to "claude", so leaving them unset is a
    # byte-for-byte no-op — today's only supported production configuration.
    # Env names are DBRAIN_-prefixed (unlike the older fields) because they
    # are operator switches set in systemd's EnvironmentFile, and a bare
    # CHAT_ENGINE is far too generic a name to claim in a service env.
    chat_engine: Literal["claude", "codex"] = Field(
        default="claude",
        validation_alias=AliasChoices("DBRAIN_CHAT_ENGINE", "chat_engine"),
        description=(
            "Engine backing the interactive chat brain. 'claude' (default) — "
            "today's tmux ClaudeSession, unchanged. 'codex' — the phase-2 "
            "CodexExecDriver, now real. The default is still 'claude': a "
            "lever that exists is not a lever that has been pulled."
        ),
    )
    codex_model: str = Field(
        default="",
        validation_alias=AliasChoices("DBRAIN_CODEX_MODEL", "codex_model"),
        description=(
            "Model for the codex engine (empty = Codex CLI default). "
            "Separate from claude_model because the catalogs share no names "
            "and are account-scoped: the phase-0 spike hit 'model is not "
            "supported when using Codex with a ChatGPT account' by passing a "
            "plausible-looking id. Leaving this empty is the safe default."
        ),
    )
    codex_sandbox: Literal["", "read-only", "workspace-write", "danger-full-access"] = (
        Field(
            default="",
            validation_alias=AliasChoices("DBRAIN_CODEX_SANDBOX", "codex_sandbox"),
            description=(
                "Sandbox for NEW codex threads (empty = the driver's built-in "
                "default, unchanged). Installers set it from the chosen "
                "permission profile: 'workspace-write' for the standard "
                "profile, 'danger-full-access' for a dedicated server."
            ),
        )
    )
    cron_engine: Literal["claude", "codex"] = Field(
        default="claude",
        validation_alias=AliasChoices("DBRAIN_CRON_ENGINE", "cron_engine"),
        description=(
            "Engine backing the isolated cron brain. Same values as "
            "chat_engine; this is the one the Codex pilot flips first, "
            "because cron jobs are stateless and the blast radius is an odd "
            "morning digest instead of a lost user reply."
        ),
    )

    # ── cron (scheduled jobs in the second brain session) ────────────
    cron_enabled: bool = Field(
        default=True,
        description="Run the in-bot cron ticker",
    )
    cron_tick_seconds: float = Field(
        default=60.0,
        description="Ticker interval; jobs.json is re-read every tick",
    )
    cron_job_timeout: float = Field(
        default=600.0,
        description="Per-job ask() timeout in the cron session",
    )
    cron_max_consecutive_errors: int = Field(
        default=3,
        description="Consecutive failures before a job is auto-disabled",
    )
    cron_retry_seconds: float = Field(
        default=300.0,
        description="Retry delay for a failed one-shot ('at') job",
    )

    # ── durable inbox ──────────
    inbox_replay_max_age: float = Field(
        default=3600.0,
        validation_alias=AliasChoices(
            "DBRAIN_INBOX_REPLAY_MAX_AGE", "inbox_replay_max_age"
        ),
        description=(
            "How old (seconds) an accepted-but-unanswered incoming message "
            "may be and still be replayed at startup. Default one hour, the "
            "same window the reference agent uses: long enough that a "
            "restart or a crash never eats a message, short enough that "
            "booting after a night down does not answer yesterday's chat. "
            "Anything older is moved to runtime_dir/inbox/stale/ instead of "
            "being replayed — nothing is deleted. 0 (or negative) disables "
            "replay entirely and retires the whole backlog."
        ),
    )

    # ── graceful stop ──────────
    shutdown_grace_seconds: float = Field(
        default=300.0,
        validation_alias=AliasChoices(
            "DBRAIN_SHUTDOWN_GRACE", "shutdown_grace_seconds"
        ),
        description=(
            "How long (seconds) a stop waits for the turns already in flight "
            "before it leaves. On SIGTERM the bot stops TAKING work — new "
            "messages are still accepted to disk and answered with a short "
            "'перезапускаюсь' line, then replayed after the restart — while "
            "whatever was already running gets this long to produce its "
            "reply. Only ever paid when something is actually running: an "
            "idle bot stops at once. A second signal exits immediately.\n\n"
            "MUST stay below the unit's TimeoutStopSec (360s in "
            "deploy/dbrain-bot.service and deploy/systemd/dbrain-bot@.service) "
            "with room for shutdown.FINAL_DRAIN_TIMEOUT on top, or systemd "
            "SIGKILLs the process mid-grace and the wait buys nothing. "
            "tests/test_shutdown.py pins that relationship."
        ),
    )

    # ── per-chat queue ─────────
    chat_queue_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("DBRAIN_CHAT_QUEUE", "chat_queue_enabled"),
        description=(
            "One turn per chat at a time; a message that arrives while the "
            "main session is working is acknowledged at once ('принял, отвечу "
            "следом') and waits ON DISK until the session is free. False → "
            "byte-for-byte the pre-queue behavior, where the same message got "
            "a brush-off or an answer from the context-less duty session. The "
            "rollback switch for this feature."
        ),
    )
    chat_queue_max_waiting: int = Field(
        default=10,
        validation_alias=AliasChoices(
            "DBRAIN_CHAT_QUEUE_MAX", "chat_queue_max_waiting"
        ),
        description=(
            "How many messages ONE chat may have waiting. Past it the sender "
            "is told plainly that the message will not be answered — nothing "
            "is ever dropped quietly. 0 (or negative) means no limit, which "
            "is only sane if you never want to be told the queue is running "
            "away from you."
        ),
    )
    chat_queue_max_age: float = Field(
        default=7200.0,
        validation_alias=AliasChoices(
            "DBRAIN_CHAT_QUEUE_MAX_AGE", "chat_queue_max_age"
        ),
        description=(
            "How long (seconds) a message may wait before the queue gives up "
            "on it, moves it to runtime_dir/chat-queue/stale/ and says so. "
            "Two hours is far beyond any legitimate turn (chat_turn_timeout "
            "is 1500s, the watchdog closes an unattended run at 1800s), so "
            "reaching it means something is wrong rather than slow. 0 (or "
            "negative) disables the age cap entirely."
        ),
    )

    # ── duty session ───────────────
    # The THIRD engine session, modelled on the cron one: same persona, same
    # vault, own session name and own runtime dir. It exists for exactly one
    # case — a Telegram message arriving while the main brain is mid
    # UNATTENDED long turn (ask-lock free, pane busy), which used to get the
    # "🛠 идёт длинная фоновая задача" brush-off and nothing else. The cron
    # session cannot be reused for this: it is /clear-ed after every job and
    # may itself be running one.
    duty_session_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("DBRAIN_DUTY_SESSION", "duty_session_enabled"),
        description=(
            "Answer from the duty session when the main one is busy. False → "
            "byte-for-byte today's behavior (the busy/maintenance brush-off "
            "message), the rollback switch for this feature."
        ),
    )
    duty_turn_timeout: float = Field(
        default=600.0,
        description=(
            "ask() timeout for a duty turn. Deliberately far below the main "
            "session's: a duty turn is contracted to be short, and the user "
            "is already waiting out a busy main session. Raised 300 → 600 "
            "(blind review F7, 2026-09-20): the FIRST duty turn of a busy "
            "window is not just the answer — it is a cold `claude` start "
            "plus the CLAUDE.md session bootstrap plus the answer, all "
            "inside this one budget, and 300s could not reliably hold all "
            "three. A warm turn is unaffected: it finishes when it "
            "finishes, this is only the ceiling."
        ),
    )
    duty_stall_timeout: float = Field(
        default=240.0,
        description=(
            "Hang detector for a DUTY turn, per-session rather than the "
            "engine default (claude_session.DEFAULT_STALL_TIMEOUT = 900s). "
            "MUST be strictly less than duty_turn_timeout, and the margin "
            "matters: ask()'s loop is bounded by `deadline = clock + "
            "timeout` while the stall check fires on `clock - last_active > "
            "stall_timeout`, so a stall timeout at or above the turn budget "
            "can never fire and the Escape is dead code (review round 3, "
            "R1 — the same defect F3 fixed for the chat path). It bites "
            "HARDER here: duty_dir is watched by nobody. The watchdog polls "
            "settings.runtime_dir only, so neither its recovery nor its "
            "orphan-reply poller ever looks at the duty session — a wedged "
            "duty pane does not heal itself and its late reply never "
            "arrives. Self-interruption inside the budget is the only "
            "backstop that path has. 240s leaves 360s of headroom under the "
            "600s budget: enough for the Escape to land and the turn to "
            "report a status."
        ),
    )
    duty_idle_reset_seconds: float = Field(
        default=3600.0,
        description=(
            "If the duty session has not been used for longer than this, it "
            "gets a /clear before the next turn. Keeps continuity inside one "
            "busy window without letting its context grow for months. 0 → "
            "clear before every duty turn."
        ),
    )
    chat_turn_timeout: float = Field(
        default=1500.0,
        description=(
            "Hard ceiling for a CHAT-initiated main-session turn. Before this "
            "existed the chat path inherited DEFAULT_TIMEOUT (3600s) — an "
            "hour of a Telegram user staring at a typing indicator. On "
            "expiry ask() keeps the request in flight (the rid is NOT marked "
            "handled), so a late reply still arrives via the watchdog's "
            "orphan path (Claude engine; under Codex the turn's process is "
            "terminated instead — see chat_session._turn_limit_keeps_inflight). "
            "0 → the old DEFAULT_TIMEOUT behavior.\n\n"
            "MUST be strictly greater than claude_session.DEFAULT_STALL_TIMEOUT "
            "(900s). ask()'s loop is bounded by `deadline = clock + timeout` "
            "while its hang detector fires on `clock - last_active > "
            "stall_timeout`; at equal values that strict inequality can never "
            "hold, so the stall interrupt becomes dead code on every chat turn "
            "and a wedged pane rides the ceiling instead of being interrupted. "
            "The default was exactly 900.0 until the blind review caught it "
            "(F3, 2026-09-20); 1500 leaves a real 600s stall window while "
            "staying well under DEFAULT_TIMEOUT. A misconfigured value is "
            "WARNED about at startup, never raised — a bot that refuses to "
            "boot is worse than one with a dead stall check."
        ),
    )

    @field_validator("runtime_dir", "vault_path", mode="after")
    @classmethod
    def _expand_user(cls, v: Path) -> Path:
        # pydantic-settings keeps "~" literal; the cron CLI expanduser-s —
        # expand here too or the bot and CLI split into different state dirs.
        # resolve() makes the path ABSOLUTE: the brain runs `cd vault && cat
        # deploy/brain-system.md`, and a relative vault_path would make that
        # cat (and --mcp-config) resolve against the wrong cwd → persona
        # silently not loaded. One root of absoluteness for all derived paths.
        return v.expanduser().resolve()

    @model_validator(mode="after")
    def _default_project_root(self) -> "Settings":
        # Runs AFTER _expand_user, so vault_path is already absolute here.
        # Unset → vault_path.parent, byte-for-byte what runtime.py/watchdog.py
        # /doctor.py computed inline before this field existed. Set → expanded
        # and resolved on the same terms as vault_path/runtime_dir, so all
        # derived paths share one root of absoluteness.
        if self.project_root is None:
            object.__setattr__(self, "project_root", self.vault_path.parent)
        else:
            object.__setattr__(
                self, "project_root", self.project_root.expanduser().resolve()
            )
        return self

    @model_validator(mode="after")
    def _warn_on_dead_duty_stall_window(self) -> "Settings":
        """Round-3 R1's invariant, in the same shape as the chat one below:
        a ``duty_stall_timeout`` at or above ``duty_turn_timeout`` can never
        fire, so the duty session loses its only self-recovery — nothing
        else watches ``duty_dir``. Warning, never an exception, for the same
        reason as below."""
        if self.duty_turn_timeout <= 0 or self.duty_stall_timeout <= 0:
            return self
        if self.duty_stall_timeout >= self.duty_turn_timeout:
            logging.getLogger(__name__).warning(
                "DUTY_STALL_TIMEOUT=%s is not less than DUTY_TURN_TIMEOUT=%s: "
                "a duty turn's stall interrupt can never fire, and nothing "
                "else watches the duty session — a wedged duty pane will not "
                "recover on its own.",
                self.duty_stall_timeout,
                self.duty_turn_timeout,
            )
        return self

    @model_validator(mode="after")
    def _warn_on_dead_chat_stall_window(self) -> "Settings":
        """Blind review F3: a ``chat_turn_timeout`` at or below the stall
        timeout silently disables ``ask()``'s stall interrupt for every chat
        turn (see the field's description for the arithmetic).

        A WARNING, deliberately not a ValidationError: this runs on every
        ``Settings()``, including the bot's own startup, and an operator
        typo that merely weakens a hang detector must not turn into a
        service that will not boot at all. The import is local so `config`
        keeps no module-level dependency on the engine layer.
        """
        if self.chat_turn_timeout <= 0:
            return self  # 0 = "use DEFAULT_TIMEOUT (3600)", which is fine
        from d_brain.services.claude_session import DEFAULT_STALL_TIMEOUT

        if self.chat_turn_timeout <= DEFAULT_STALL_TIMEOUT:
            logging.getLogger(__name__).warning(
                "CHAT_TURN_TIMEOUT=%s is not greater than the stall timeout "
                "(%s): ask()'s stall interrupt can never fire on a chat turn, "
                "so a wedged pane will ride the full ceiling instead of being "
                "interrupted. Set it above %s.",
                self.chat_turn_timeout,
                DEFAULT_STALL_TIMEOUT,
                DEFAULT_STALL_TIMEOUT,
            )
        return self

    @model_validator(mode="after")
    def _warn_on_grace_the_unit_will_not_honour(self) -> "Settings":
        """A grace the systemd unit cannot cover buys nothing.

        The unit's ``TimeoutStopSec`` is the real ceiling: past it systemd
        SIGKILLs the process mid-grace, so an operator who raises
        ``DBRAIN_SHUTDOWN_GRACE`` in ``/etc/dbrain/<instance>/.env`` — the same
        file systemd reads as ``EnvironmentFile=``, which is exactly why this
        is easy to get wrong — silently gets the old hard kill back (blind
        review 6). A WARNING, not an error, for the same reason as the check
        above: a stop that is less graceful than intended must not become a
        bot that will not boot.
        """
        from d_brain.services.shutdown import STOP_OVERHEAD, UNIT_TIMEOUT_STOP_SEC

        if self.shutdown_grace_seconds + STOP_OVERHEAD > UNIT_TIMEOUT_STOP_SEC:
            logging.getLogger(__name__).warning(
                "DBRAIN_SHUTDOWN_GRACE=%s plus %.0fs of stop overhead exceeds "
                "the unit's TimeoutStopSec=%.0f: systemd will SIGKILL the "
                "process before the grace is up, so a turn in flight is cut "
                "off exactly as it was before. Lower the grace to %.0f or "
                "raise TimeoutStopSec in the unit.",
                self.shutdown_grace_seconds,
                STOP_OVERHEAD,
                UNIT_TIMEOUT_STOP_SEC,
                UNIT_TIMEOUT_STOP_SEC - STOP_OVERHEAD,
            )
        return self

    @property
    def cron_dir(self) -> Path:
        """Cron state dir: jobs.json + the cron session's runtime files."""
        return self.runtime_dir / "cron"

    @property
    def duty_dir(self) -> Path:
        """Duty state dir: the duty session's own runtime files + last-use
        stamp. Sibling of cron_dir, for the same isolation reason."""
        return self.runtime_dir / "duty"

    @property
    def admin_chat_id(self) -> int | None:
        """First allowed user — destination for health alerts / reports."""
        return self.allowed_user_ids[0] if self.allowed_user_ids else None

    @property
    def daily_path(self) -> Path:
        """Path to daily notes directory."""
        return self.vault_path / "daily"

    @property
    def attachments_path(self) -> Path:
        """Path to attachments directory."""
        return self.vault_path / "attachments"

    @property
    def thoughts_path(self) -> Path:
        """Path to thoughts directory."""
        return self.vault_path / "thoughts"


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()
