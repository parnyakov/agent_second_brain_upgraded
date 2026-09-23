"""Tests for the extended Settings (tmux-session fields)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

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


def test_duty_fields_have_safe_defaults():
    """on by default, but every knob is a rollback."""
    s = _settings()
    assert s.duty_session_enabled is True
    assert s.duty_turn_timeout == 600.0
    assert s.duty_stall_timeout == 240.0
    assert s.duty_idle_reset_seconds == 3600.0
    assert s.chat_turn_timeout == 1500.0


def test_duty_stall_timeout_can_actually_fire_on_production_defaults(caplog):
    """Review round 3, R1 — the same inequality F3 pinned for chat, now for
    the duty session, and on the SHIPPED defaults rather than a hand-made
    Settings.

    ask()'s loop is bounded by `deadline = clock + duty_turn_timeout` while
    the stall check fires on `clock - last_active > stall_timeout`, so a
    stall timeout at or above the budget is unreachable. The engine default
    (900s) was above the 600s duty budget, which made the Escape dead code —
    and duty_dir is watched by nobody: the watchdog polls runtime_dir only,
    so a wedged duty pane neither heals nor delivers a late reply.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="d_brain.config"):
        s = _settings()
    assert s.duty_stall_timeout < s.duty_turn_timeout
    assert not caplog.records  # the shipped defaults never warn


def test_a_dead_duty_stall_window_warns_but_still_boots(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="d_brain.config"):
        s = _settings(duty_stall_timeout=600.0, duty_turn_timeout=600.0)
    assert s.duty_stall_timeout == 600.0  # accepted, not raised
    assert any("DUTY_STALL_TIMEOUT" in r.getMessage() for r in caplog.records)


def test_chat_turn_timeout_leaves_the_stall_interrupt_alive(caplog):
    """Blind review F3 (blocker) — an INEQUALITY on the PRODUCTION default,
    not on a hand-made Settings.

    ask()'s loop is bounded by `deadline = clock + timeout` while its hang
    detector fires on `clock - last_active > stall_timeout`. At equal values
    that strict inequality can never hold: the stall interrupt becomes dead
    code on every chat turn and a wedged pane rides the full ceiling instead
    of being interrupted. The default WAS exactly DEFAULT_STALL_TIMEOUT.
    """
    import logging

    from d_brain.services.claude_session import DEFAULT_STALL_TIMEOUT, DEFAULT_TIMEOUT

    with caplog.at_level(logging.WARNING, logger="d_brain.config"):
        s = _settings()
    assert s.chat_turn_timeout > DEFAULT_STALL_TIMEOUT
    assert s.chat_turn_timeout < DEFAULT_TIMEOUT
    assert not caplog.records  # the shipped default never warns


def test_a_dead_stall_window_warns_but_still_boots(caplog):
    """A misconfigured value only weakens a hang detector. Refusing to start
    the bot over it would be strictly worse, so this is a log warning and
    never an exception."""
    import logging

    from d_brain.services.claude_session import DEFAULT_STALL_TIMEOUT

    with caplog.at_level(logging.WARNING, logger="d_brain.config"):
        s = _settings(chat_turn_timeout=DEFAULT_STALL_TIMEOUT)
    assert s.chat_turn_timeout == DEFAULT_STALL_TIMEOUT  # accepted, not raised
    assert any("stall" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="d_brain.config"):
        _settings(chat_turn_timeout=0.0)  # 0 = "use DEFAULT_TIMEOUT" — fine
    assert not caplog.records


def test_duty_dir_lives_under_runtime_dir():
    s = _settings(runtime_dir=Path("/tmp/rt"))
    assert s.duty_dir == s.runtime_dir / "duty"
    assert s.duty_dir != s.cron_dir  # never shares state with the cron brain


def test_duty_switch_reads_its_operator_env_name(monkeypatch):
    """The rollback lever is set in systemd's EnvironmentFile, so it must
    answer to the DBRAIN_-prefixed name the other operator switches use."""
    monkeypatch.setenv("DBRAIN_DUTY_SESSION", "false")
    assert _settings().duty_session_enabled is False
    monkeypatch.setenv("DBRAIN_DUTY_SESSION", "true")
    assert _settings().duty_session_enabled is True


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
    exact ".env" when it's unset, so an existing --user unit (which never
    sets it) is unaffected."""
    monkeypatch.delenv("DBRAIN_ENV_FILE", raising=False)
    import importlib

    import d_brain.config as config_module

    importlib.reload(config_module)
    assert config_module.Settings.model_config["env_file"] == ".env"

    instance_env = tmp_path / "instance.env"
    instance_env.write_text(
        "TELEGRAM_BOT_TOKEN=instance-token\nDEEPGRAM_API_KEY=instance-key\n"
    )
    monkeypatch.setenv("DBRAIN_ENV_FILE", str(instance_env))
    importlib.reload(config_module)
    try:
        assert config_module.Settings.model_config["env_file"] == str(instance_env)
        s = config_module.Settings()
        assert s.telegram_bot_token == "instance-token"
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


def _base_settings(tmp_path, **over):
    kwargs = dict(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path / "vault",
        _env_file=None,
    )
    kwargs.update(over)
    return Settings(**kwargs)


def test_long_run_max_seconds_defaults_above_the_alert_threshold(tmp_path):
    """the three long-run thresholds form a ladder — alert (a
    heads-up) then, optionally, nudge (ask the session to self-close) then
    max (the watchdog closes it). A default that inverted that order would
    close the turn before the owner was ever told anything."""
    s = _base_settings(tmp_path)
    assert s.long_run_max_seconds == 1800.0
    assert s.long_run_max_seconds > s.long_run_alert_seconds


def test_long_run_max_seconds_zero_is_the_rollback_switch(tmp_path):
    """0 must be accepted (not rejected as "invalid"): it is the documented
    way to turn the auto-close off without touching code."""
    s = _base_settings(tmp_path, long_run_max_seconds=0)
    assert s.long_run_max_seconds == 0.0


def test_long_run_max_seconds_reads_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LONG_RUN_MAX_SECONDS", "2400")
    try:
        assert _base_settings(tmp_path).long_run_max_seconds == 2400.0
    finally:
        monkeypatch.delenv("LONG_RUN_MAX_SECONDS", raising=False)


def test_pane_geometry_defaults_match_the_old_hardcoded_values(tmp_path):
    """Setting neither variable must be a byte-for-byte no-op: 200x50 is
    what claude_session.py hardcoded before these fields existed."""
    s = _base_settings(tmp_path)
    assert (s.pane_width, s.pane_height) == (200, 50)


def test_pane_geometry_reads_the_dbrain_prefixed_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("DBRAIN_PANE_WIDTH", "160")
    monkeypatch.setenv("DBRAIN_PANE_HEIGHT", "50")
    s = _base_settings(tmp_path)
    assert (s.pane_width, s.pane_height) == (160, 50)


def test_pane_geometry_rejects_a_degenerate_size(tmp_path):
    """A pane a handful of columns wide would wrap every line into noise and
    break marker parsing outright — refuse it at config time, loudly."""
    with pytest.raises(ValidationError):
        _base_settings(tmp_path, pane_width=10)
    with pytest.raises(ValidationError):
        _base_settings(tmp_path, pane_height=2)
