"""Tests for the extended Settings (tmux-session fields)."""

from pathlib import Path

from d_brain.config import Settings


def _settings(**over):
    base = dict(telegram_bot_token="t", deepgram_api_key="d", _env_file=None)
    base.update(over)
    return Settings(**base)


def test_new_fields_have_safe_defaults():
    s = _settings()
    assert s.tz == "UTC"
    assert s.claude_model == ""  # "" → use the session's default model
    assert isinstance(s.runtime_dir, Path)


def test_admin_chat_id_is_first_allowed_user():
    s = _settings(allowed_user_ids=[111, 222])
    assert s.admin_chat_id == 111


def test_admin_chat_id_none_when_no_users():
    s = _settings(allowed_user_ids=[])
    assert s.admin_chat_id is None


def test_overrides_from_kwargs():
    s = _settings(claude_model="sonnet", tz="Asia/Tashkent")
    assert s.claude_model == "sonnet"
    assert s.tz == "Asia/Tashkent"


def test_cron_fields_have_safe_defaults():
    s = _settings()
    assert s.cron_enabled is True
    assert s.cron_tick_seconds == 60.0
    assert s.cron_job_timeout == 600.0
    assert s.cron_max_consecutive_errors == 3
    assert s.cron_retry_seconds == 300.0


def test_cron_dir_lives_under_runtime_dir():
    # Assert the invariant against the resolved runtime_dir — not a literal
    # path: the validator now resolve()s, and on macOS /tmp is a symlink to
    # /private/tmp, so a hardcoded "/tmp/rt/cron" would spuriously mismatch.
    s = _settings(runtime_dir=Path("/tmp/rt"))
    assert s.cron_dir == s.runtime_dir / "cron"


def test_tilde_paths_are_expanded():
    # pydantic-settings does not expand ~ on its own; the cron CLI does
    # expanduser — without this the bot and the CLI would silently use
    # two different state dirs.
    s = _settings(runtime_dir="~/.dbrain", vault_path="~/vault")
    assert s.runtime_dir.is_absolute()
    assert "~" not in s.runtime_dir.parts
    assert s.vault_path.is_absolute()
    assert "~" not in s.vault_path.parts


def test_relative_paths_are_resolved_absolute():
    # The default vault_path is the RELATIVE "./vault". The brain starts with
    # `cd vault && cat deploy/brain-system.md`, so a relative persona path
    # resolves against vault/ after the cd and loads NOTHING — the brain boots
    # with no persona and no reply contract. Resolving here keeps the derived
    # project_root / persona / mcp paths absolute regardless of cwd.
    s = _settings(runtime_dir="./rt", vault_path="./vault")
    assert s.runtime_dir.is_absolute()
    assert s.vault_path.is_absolute()


# ── C1: multi-instance settings default to today's exact behavior ────────


def test_multi_instance_settings_default_to_current_behavior(tmp_path):
    """Phase 1 must be a no-op until someone deliberately sets these."""
    s = Settings(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path / "vault",
        _env_file=None,
    )
    assert s.bot_unit == "dbrain-bot.service"
    assert s.systemd_scope == "user"
    assert s.project_root == s.vault_path.parent


def test_env_file_defaults_to_dotenv_and_honors_dbrain_env_file(tmp_path, monkeypatch):
    """Multi-instance units share one WorkingDirectory, so a hardcoded
    relative ".env" would always resolve to whichever instance's real file
    happens to live there. Settings must read DBRAIN_ENV_FILE at class
    definition time (systemd's own EnvironmentFile= already lands it in the
    process env before this module is imported) and fall back to today's
    exact ".env" when it's unset, so the owner's existing --user unit (which
    never sets it) is unaffected."""
    monkeypatch.delenv("DBRAIN_ENV_FILE", raising=False)
    import importlib

    import d_brain.config as config_module

    importlib.reload(config_module)
    assert config_module.Settings.model_config["env_file"] == ".env"

    instance_env = tmp_path / "second.env"
    instance_env.write_text(
        "TELEGRAM_BOT_TOKEN=second-token\nDEEPGRAM_API_KEY=second-key\n"
    )
    monkeypatch.setenv("DBRAIN_ENV_FILE", str(instance_env))
    importlib.reload(config_module)
    try:
        assert config_module.Settings.model_config["env_file"] == str(instance_env)
        s = config_module.Settings()
        assert s.telegram_bot_token == "second-token"
    finally:
        monkeypatch.delenv("DBRAIN_ENV_FILE", raising=False)
        importlib.reload(config_module)


def test_systemd_scope_rejects_unknown_values(tmp_path):
    """A typo must fail loudly at startup rather than silently picking a
    restart path that cannot work for this instance."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(
            telegram_bot_token="t",
            deepgram_api_key="d",
            vault_path=tmp_path / "vault",
            systemd_scope="sudo",
            _env_file=None,
        )
