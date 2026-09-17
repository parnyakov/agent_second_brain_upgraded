"""Application configuration using Pydantic Settings."""

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

    # ── multi-instance parameterization (C1, multi-instance plan) ──
    # Every field here defaults to EXACTLY today's single-instance behavior.
    # Setting none of them must be a byte-for-byte no-op.
    project_root: Path | None = Field(
        default=None,
        description=(
            "Code checkout root — where deploy/brain-system.md, deploy/"
            "tmux.conf and mcp-config.json live. Empty → vault_path.parent, "
            "which is exactly what the code derived before this field "
            "existed. A second instance whose vault lives OUTSIDE the "
            "checkout (e.g. /var/lib/dbrain-second/vault) sets "
            "PROJECT_ROOT to the shared checkout instead."
        ),
    )
    bot_unit: str = Field(
        default="dbrain-bot.service",
        description=(
            "systemd unit the DeliveryGuard restarts and that operator "
            "messages name. Default is today's hardcoded unit name; a "
            "templated instance sets e.g. dbrain-bot@second.service."
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
    transcript_shadow_mode: bool = Field(
        default=False,
        description=(
            "R1 (Fable audit, 2026-08-22): also extract replies from the "
            "session's JSONL transcript and log agreement/disagreement with "
            "the panel path. Diagnostic only — never changes what is "
            "actually delivered to Telegram this round."
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
            "~10x above this (1.3% below vs ~12% above, incident registry "
            "2026-08-22)."
        ),
    )
    long_run_alert_seconds: float = Field(
        default=900.0,
        description=(
            "agent-infra-backlog item 22 (2026-09): how long an UNATTENDED "
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

    # ── engine selection (Codex-engine plan, phase 1) ────────────────
    # PER-SESSION on purpose, not one global switch: the pilot runs Codex on
    # the isolated cron session while the main chat brain stays on Claude Code
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

    @property
    def cron_dir(self) -> Path:
        """Cron state dir: jobs.json + the cron session's runtime files."""
        return self.runtime_dir / "cron"

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
