"""Engine seam (Codex plan, phase 1).

Two things are worth testing here and nothing else is:

1. The plan's central claim — "ClaudeSession satisfies EngineDriver by
   duck-typing, zero wrapper code" — is a claim about real signatures, so it
   is checked against real signatures instead of being asserted in prose.
2. The hard constraint — engine="claude" (or unset) builds EXACTLY the object
   the pre-seam code built, with exactly the same constructor arguments.
"""

import inspect

import pytest

import d_brain.services.runtime as rt
from d_brain.config import Settings
from d_brain.services.claude_session import ClaudeSession
from d_brain.services.codex_driver import CodexExecDriver
from d_brain.services.engine import EngineDriver

# Every method the Protocol declares. Listed explicitly (not derived from the
# Protocol) so that adding a method to EngineDriver without checking it
# against ClaudeSession fails this test.
_PROTOCOL_METHODS = (
    "ask",
    "is_turn_active",
    "is_pane_turn_active",
    "is_steerable_turn",
    "steer",
    "interrupt",
    "send_control",
    "is_working",
    "force_recover",
    "ensure_session",
    "is_healthy",
    "kill",
    "capture_text",
    "last_reply_for_resend",
    "pop_orphan_replies",
)


def _settings(tmp_path, **over):
    base = dict(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path / "vault",
        runtime_dir=tmp_path / "rt",
        _env_file=None,
    )
    base.update(over)
    deploy = tmp_path / "deploy"
    deploy.mkdir(parents=True, exist_ok=True)
    (deploy / "brain-system.md").write_text("# d-brain session contract\n")
    return Settings(**base)


# ── 1. the duck-typing claim ─────────────────────────────────────────────


def test_protocol_lists_exactly_the_methods_under_test():
    declared = {
        name
        for name, obj in vars(EngineDriver).items()
        if not name.startswith("_") and inspect.isfunction(obj)
    }
    assert declared == set(_PROTOCOL_METHODS)


@pytest.mark.parametrize("name", _PROTOCOL_METHODS)
def test_claude_session_matches_protocol_signature(name):
    """No adapter is needed only if every signature already lines up —
    same parameter names, kinds, defaults and annotations."""
    assert hasattr(ClaudeSession, name), f"ClaudeSession is missing {name}()"
    got = inspect.signature(getattr(ClaudeSession, name))
    want = inspect.signature(getattr(EngineDriver, name))
    assert got == want, f"{name}(): {got} != protocol {want}"


def test_claude_session_is_an_engine_driver_instance(tmp_path):
    """runtime_checkable isinstance only proves method presence — the
    signature test above is the real check. Kept because this is the
    assertion phase 2's CodexExecDriver will be held to as well."""
    sess = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path,
        runtime_dir=tmp_path / "rt",
    )
    assert isinstance(sess, EngineDriver)


# ── 2. engine="claude" builds exactly the pre-seam object ────────────────


# The constructor call runtime._build_session() made BEFORE the engine
# parameter existed, transcribed from that revision. Any drift in the claude
# branch — an added kwarg, a dropped one, a positional argument, a different
# value — breaks this.
def _expected_kwargs(settings, *, session_name, runtime_dir):
    return dict(
        session_name=session_name,
        work_dir=settings.vault_path,
        runtime_dir=runtime_dir,
        mcp_config=None,  # mcp-config.json does not exist under tmp_path
        system_prompt_file=settings.project_root / "deploy" / "brain-system.md",
        model=settings.claude_model or None,
        tmux_config=None,  # deploy/tmux.conf does not exist under tmp_path
        # Pane geometry became settable 2026-09-20; the defaults are the
        # values that used to be hardcoded in claude_session.py, so the
        # session this builds is still the pre-seam one.
        pane_width=settings.pane_width,
        pane_height=settings.pane_height,
    )


@pytest.fixture
def recorded(monkeypatch):
    """Capture the ClaudeSession constructor call runtime.py makes."""
    calls = []

    class _Recorder(ClaudeSession):
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(rt, "ClaudeSession", _Recorder)
    return calls


@pytest.mark.parametrize("engine", [None, "claude"])
def test_claude_engine_builds_the_same_session_as_before(engine, recorded, tmp_path):
    rt.reset()
    settings = _settings(tmp_path)
    kwargs = {} if engine is None else {"engine": engine}
    sess = rt._build_session(
        settings,
        session_name="dbrain_fixed",
        runtime_dir=settings.runtime_dir,
        **kwargs,
    )

    assert isinstance(sess, ClaudeSession)
    assert len(recorded) == 1
    args, got = recorded[0]
    assert args == ()  # everything was passed by keyword, as before
    assert got == _expected_kwargs(
        settings, session_name="dbrain_fixed", runtime_dir=settings.runtime_dir
    )


def test_unset_config_routes_both_sessions_through_the_claude_branch(
    recorded, tmp_path
):
    """The production default: neither env var set ⇒ chat AND cron build
    ClaudeSession with the very same arguments as before the seam."""
    rt.reset()
    settings = _settings(tmp_path)
    assert settings.chat_engine == "claude"
    assert settings.cron_engine == "claude"

    chat = rt.get_session(settings)
    cron = rt.get_cron_session(settings)
    rt.reset()

    assert isinstance(chat, ClaudeSession) and isinstance(cron, ClaudeSession)
    assert [a for a, _ in recorded] == [(), ()]
    assert recorded[0][1] == _expected_kwargs(
        settings,
        session_name=chat.session_name,
        runtime_dir=settings.runtime_dir,
    )
    assert recorded[1][1] == _expected_kwargs(
        settings,
        session_name=f"{chat.session_name}_cron",
        runtime_dir=settings.cron_dir,
    )


# ── 3. the codex branch ──────────────────────────────────────────────────
#
# Phase 2 replaced the NotImplementedError stub with a real driver. What is
# asserted here is only the seam: the branch builds a CodexExecDriver, and it
# still refuses loudly (never falls back to Claude) when its persona file is
# missing. The driver's own behavior lives in tests/test_codex_driver.py.


def test_codex_engine_builds_the_codex_driver(tmp_path):
    rt.reset()
    settings = _settings(tmp_path)
    (settings.project_root / "deploy" / "codex-agents.md").write_text(
        "# d-brain codex agent contract\n"
    )
    sess = rt._build_session(
        settings,
        session_name="x",
        runtime_dir=settings.runtime_dir,
        engine="codex",
    )
    assert isinstance(sess, CodexExecDriver)
    assert isinstance(sess, EngineDriver)
    assert sess.work_dir == settings.vault_path
    assert sess.runtime_dir == settings.runtime_dir


def test_codex_engine_refuses_a_missing_persona_instead_of_falling_back(tmp_path):
    """A silent fallback to Claude would make an operator think the pilot was
    running when it was not — the stub's original point, preserved."""
    rt.reset()
    settings = _settings(tmp_path)
    with pytest.raises(RuntimeError, match="codex persona file"):
        rt._build_session(
            settings,
            session_name="x",
            runtime_dir=settings.runtime_dir,
            engine="codex",
        )


def test_unknown_engine_is_rejected(tmp_path):
    rt.reset()
    settings = _settings(tmp_path)
    with pytest.raises(ValueError, match="unknown engine"):
        rt._build_session(
            settings,
            session_name="x",
            runtime_dir=settings.runtime_dir,
            engine="gpt",
        )


# ── 4. config plumbing ───────────────────────────────────────────────────


def test_engine_settings_default_to_claude(tmp_path):
    settings = _settings(tmp_path)
    assert settings.chat_engine == "claude"
    assert settings.cron_engine == "claude"


def test_engine_settings_read_dbrain_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("DBRAIN_CHAT_ENGINE", "codex")
    monkeypatch.setenv("DBRAIN_CRON_ENGINE", "codex")
    settings = _settings(tmp_path)
    assert settings.chat_engine == "codex"
    assert settings.cron_engine == "codex"


def test_engine_settings_are_independent_per_session(tmp_path, monkeypatch):
    """The pilot's whole shape: cron on Codex while chat stays on Claude."""
    monkeypatch.setenv("DBRAIN_CRON_ENGINE", "codex")
    settings = _settings(tmp_path)
    assert settings.chat_engine == "claude"
    assert settings.cron_engine == "codex"


def test_engine_settings_reject_unknown_values(tmp_path):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _settings(tmp_path, chat_engine="gpt")


def test_codex_sandbox_is_passed_only_when_configured(tmp_path):
    """The permission profile chosen at install time reaches new codex
    threads; an unset value keeps the driver's own default untouched."""
    from d_brain.services import codex_driver

    rt.reset()
    settings = _settings(tmp_path)
    (settings.project_root / "deploy" / "codex-agents.md").write_text(
        "# d-brain codex agent contract\n"
    )
    default = rt._build_session(
        settings, session_name="x", runtime_dir=settings.runtime_dir, engine="codex"
    )
    assert default.sandbox == codex_driver.DEFAULT_SANDBOX

    rt.reset()
    configured = _settings(tmp_path, codex_sandbox="danger-full-access")
    sess = rt._build_session(
        configured, session_name="x", runtime_dir=configured.runtime_dir, engine="codex"
    )
    assert sess.sandbox == "danger-full-access"
