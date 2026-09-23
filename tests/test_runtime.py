"""Tests for the shared session/processor singletons."""

import subprocess

import d_brain.services.runtime as rt
from d_brain.config import Settings


def _settings(tmp_path, *, persona: bool = True, **over):
    base = dict(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path / "vault",
        runtime_dir=tmp_path / "rt",
        _env_file=None,
    )
    base.update(over)
    if persona:  # the boot assertion requires the persona file
        deploy = tmp_path / "deploy"
        deploy.mkdir(parents=True, exist_ok=True)
        (deploy / "brain-system.md").write_text("# d-brain session contract\n")
    return Settings(**base)


def test_get_session_is_singleton(tmp_path):
    rt.reset()
    s = _settings(tmp_path)
    assert rt.get_session(s) is rt.get_session(s)


def test_get_processor_is_singleton_and_wired_to_session(tmp_path):
    rt.reset()
    s = _settings(tmp_path)
    p = rt.get_processor(s)
    assert rt.get_processor(s) is p
    assert p.session is rt.get_session(s)


def test_session_name_persisted_and_stable(tmp_path):
    rt.reset()
    s = _settings(tmp_path)
    name1 = rt.get_session(s).session_name
    rt.reset()  # drop in-memory singleton; must re-read persisted name
    name2 = rt.get_session(s).session_name
    assert name1 == name2
    assert name1.startswith("dbrain")


def test_explicit_session_name_used(tmp_path):
    rt.reset()
    s = _settings(tmp_path, brain_session_name="dbrain_fixed")
    assert rt.get_session(s).session_name == "dbrain_fixed"


def test_get_cron_session_is_isolated_sibling(tmp_path):
    """The cron brain is a SECOND ClaudeSession: same persona and vault,
    but its own tmux session name and its own runtime dir, so pane.lock /
    pane.log / ready never collide with the main brain's."""
    rt.reset()
    s = _settings(tmp_path)
    main = rt.get_session(s)
    cron = rt.get_cron_session(s)
    assert cron is not main
    assert cron.session_name == f"{main.session_name}_cron"
    assert cron.runtime_dir == s.cron_dir
    assert cron.runtime_dir != main.runtime_dir
    assert cron.work_dir == main.work_dir


def test_engine_round_trip_keeps_names_and_dirs(tmp_path):
    """The owner flips DBRAIN_CHAT_ENGINE/DBRAIN_CRON_ENGINE and restarts the
    bot. Claude → Codex → Claude must land on
    the SAME persisted tmux names and runtime dirs, so the Claude side finds
    its own sessions again (exact-match addressing and the foreign-view check
    in ClaudeSession take it from there) and the Codex side its own thread."""
    from d_brain.services.claude_session import ClaudeSession, exact_target
    from d_brain.services.codex_driver import CodexExecDriver

    def build(engine):
        rt.reset()
        s = _settings(tmp_path, chat_engine=engine, cron_engine=engine)
        (tmp_path / "deploy" / "codex-agents.md").write_text(
            "# d-brain codex agent contract\n"
        )
        return s, rt.get_session(s), rt.get_cron_session(s)

    s, chat1, cron1 = build("claude")
    # The Claude brains were running before the switch: pinned transcript ids.
    for sess, sid in ((chat1, "chat-sid"), (cron1, "cron-sid")):
        (sess.runtime_dir / "session_id").write_text(sid + "\n")
    _, chat_cx, cron_cx = build("codex")
    _, chat2, cron2 = build("claude")
    rt.reset()

    assert isinstance(chat1, ClaudeSession) and isinstance(chat2, ClaudeSession)
    assert isinstance(chat_cx, CodexExecDriver)
    assert isinstance(cron_cx, CodexExecDriver)
    for a, b in ((chat1, chat_cx), (chat1, chat2), (cron1, cron_cx), (cron1, cron2)):
        assert a.session_name == b.session_name
        assert a.runtime_dir == b.runtime_dir
    assert cron2.session_name == f"{chat2.session_name}_cron"
    # The main brain can never resolve to its cron sibling.
    assert chat2._target == exact_target(chat2.session_name)
    assert cron2._target == exact_target(cron2.session_name)
    assert chat2._target != cron2._target

    # Back on Claude, the still-running tmux sessions show their own
    # conversations: ensure_session() reuses them as they are — no new
    # process, no keys, and the pinned session ids survive the round trip.
    ready = (
        "────────────────────\n❯\n────────────────────\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    for sess, sid in ((chat2, "chat-sid"), (cron2, "cron-sid")):
        calls = []

        def runner(args, _calls=calls, **kwargs):  # noqa: ANN001
            _calls.append(args[1])
            out = ready if args[1] == "capture-pane" else "200x50\n"
            return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

        sess._runner = runner
        sess.ensure_session()
        assert "new-session" not in calls and "send-keys" not in calls
        assert (sess.runtime_dir / "session_id").read_text().strip() == sid


def test_get_duty_session_is_an_isolated_third_sibling(tmp_path):
    """the duty brain is a THIRD session — same persona
    and vault, its own session name and its own runtime dir, sharing state
    with neither the main brain nor the cron one."""
    rt.reset()
    s = _settings(tmp_path)
    main = rt.get_session(s)
    cron = rt.get_cron_session(s)
    duty = rt.get_duty_session(s)
    assert duty is not main and duty is not cron
    assert duty.session_name == f"{main.session_name}_duty"
    assert duty.runtime_dir == s.duty_dir
    assert duty.runtime_dir not in (main.runtime_dir, cron.runtime_dir)
    assert duty.work_dir == main.work_dir


def test_get_duty_session_follows_the_chat_engine(tmp_path):
    """It stands in for the CHAT brain, so flipping DBRAIN_CHAT_ENGINE must
    take it along — otherwise an operator on Codex would still get a tmux
    Claude session started behind their back. cron_engine must not move it."""
    from d_brain.services.claude_session import ClaudeSession
    from d_brain.services.codex_driver import CodexExecDriver

    (tmp_path / "deploy").mkdir(parents=True, exist_ok=True)
    (tmp_path / "deploy" / "codex-agents.md").write_text(
        "# d-brain codex agent contract\n"
    )
    rt.reset()
    duty = rt.get_duty_session(_settings(tmp_path, chat_engine="codex"))
    assert isinstance(duty, CodexExecDriver)
    rt.reset()
    duty = rt.get_duty_session(_settings(tmp_path, cron_engine="codex"))
    assert isinstance(duty, ClaudeSession)
    rt.reset()


def test_duty_session_gets_a_stall_timeout_inside_its_own_budget(tmp_path):
    """Review round 3, R1: the engine default (900s) sits above the duty
    session's whole turn budget (600s), so its stall interrupt could never
    fire — and duty_dir is watched by nobody (the watchdog polls
    runtime_dir only), so a wedged duty pane would neither heal itself nor
    ever deliver its reply. Self-interruption inside the budget is the only
    backstop that session has."""
    from d_brain.services.claude_session import DEFAULT_STALL_TIMEOUT

    rt.reset()
    s = _settings(tmp_path)
    duty = rt.get_duty_session(s)
    rt.reset()
    assert duty._stall_timeout == s.duty_stall_timeout
    assert duty._stall_timeout < s.duty_turn_timeout  # the whole point
    assert duty._stall_timeout < DEFAULT_STALL_TIMEOUT


def test_only_the_duty_session_overrides_the_stall_timeout(tmp_path, monkeypatch):
    """The main and cron sessions must be constructed with byte-for-byte the
    arguments they were before this parameter existed — `stall_timeout` must
    not reach their constructor at all, so whatever the driver's own default
    is (including a future change to it) stays theirs."""
    from d_brain.services.claude_session import DEFAULT_STALL_TIMEOUT

    seen: list[dict] = []

    class Recorder:
        def __init__(self, **kwargs):
            seen.append(kwargs)
            self.session_name = kwargs["session_name"]
            self.runtime_dir = kwargs["runtime_dir"]

    monkeypatch.setattr(rt, "ClaudeSession", Recorder)
    rt.reset()
    s = _settings(tmp_path)
    rt.get_session(s)
    rt.get_cron_session(s)
    rt.get_duty_session(s)
    rt.reset()

    main, cron, duty = seen
    assert "stall_timeout" not in main
    assert "stall_timeout" not in cron
    assert duty["stall_timeout"] == s.duty_stall_timeout < DEFAULT_STALL_TIMEOUT


def test_the_codex_duty_session_gets_the_same_override(tmp_path):
    """Engine switching stays symmetric: the hole R1 closes is the duty
    session's, not the tmux driver's."""
    from d_brain.services.codex_driver import CodexExecDriver

    (tmp_path / "deploy").mkdir(parents=True, exist_ok=True)
    (tmp_path / "deploy" / "codex-agents.md").write_text(
        "# d-brain codex agent contract\n"
    )
    rt.reset()
    s = _settings(tmp_path, chat_engine="codex")
    duty = rt.get_duty_session(s)
    rt.reset()
    assert isinstance(duty, CodexExecDriver)
    assert duty._stall_timeout == s.duty_stall_timeout


def test_a_zero_duty_stall_timeout_falls_back_to_the_engine_default(tmp_path):
    """0 is the rollback: no override, exactly the pre-R1 construction."""
    from d_brain.services.claude_session import DEFAULT_STALL_TIMEOUT

    rt.reset()
    duty = rt.get_duty_session(_settings(tmp_path, duty_stall_timeout=0.0))
    rt.reset()
    assert duty._stall_timeout == DEFAULT_STALL_TIMEOUT


def test_get_duty_session_is_singleton_and_reset_clears(tmp_path):
    rt.reset()
    s = _settings(tmp_path)
    d1 = rt.get_duty_session(s)
    assert rt.get_duty_session(s) is d1
    rt.reset()
    assert rt.get_duty_session(s) is not d1


def test_get_duty_session_refuses_without_persona(tmp_path):
    import pytest

    rt.reset()
    s = _settings(tmp_path, persona=False)
    with pytest.raises(RuntimeError, match="persona"):
        rt.get_duty_session(s)


def test_get_cron_session_is_singleton_and_reset_clears(tmp_path):
    rt.reset()
    s = _settings(tmp_path)
    c1 = rt.get_cron_session(s)
    assert rt.get_cron_session(s) is c1
    rt.reset()
    assert rt.get_cron_session(s) is not c1


def test_get_cron_session_refuses_without_persona(tmp_path):
    import pytest

    rt.reset()
    s = _settings(tmp_path, persona=False)
    with pytest.raises(RuntimeError, match="persona"):
        rt.get_cron_session(s)


def test_relative_vault_path_yields_absolute_brain_paths(tmp_path, monkeypatch):
    """A fork installed with the default relative VAULT_PATH=./vault must
    still get ABSOLUTE paths into the start command. The brain runs
    `cd vault && cat deploy/brain-system.md`: a relative persona path would
    resolve against vault/ AFTER the cd and load nothing — silently booting a
    personality-less agent (the boot assertion passes because it reads from
    the bot's cwd, not the brain's). Resolving vault_path at config time keeps
    project_root / persona / mcp absolute regardless of the brain's cwd."""
    rt.reset()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "vault").mkdir()
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    (deploy / "brain-system.md").write_text("# d-brain session contract\n")
    s = Settings(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path="./vault",
        runtime_dir="./rt",
        _env_file=None,
    )
    sess = rt.get_session(s)
    assert sess.system_prompt_file is not None
    assert sess.system_prompt_file.is_absolute()
    assert sess.work_dir.is_absolute()


def test_get_session_refuses_without_persona(tmp_path):
    """runtime.py used to silently pass system_prompt_file=None when the
    persona file is missing — booting a personality-less vanilla agent.
    v3.0: refuse loudly instead."""
    import pytest

    rt.reset()
    s = _settings(tmp_path, persona=False)
    with pytest.raises(RuntimeError, match="persona"):
        rt.get_session(s)


# ── C1/C2: project_root (multi-instance parameterization) ────────────────


def test_project_root_defaults_to_vault_parent(tmp_path):
    """Unset PROJECT_ROOT must reproduce the pre-C2 behavior EXACTLY: the
    persona/mcp/tmux paths were derived from vault_path.parent inline."""
    rt.reset()
    s = _settings(tmp_path)
    assert s.project_root == s.vault_path.parent == tmp_path
    sess = rt.get_session(s)
    assert sess.system_prompt_file == tmp_path / "deploy" / "brain-system.md"


def test_project_root_override_moves_persona_lookup(tmp_path):
    """A second instance keeps its vault OUTSIDE the code checkout, so the
    persona must be looked up under PROJECT_ROOT, not next to the vault."""
    rt.reset()
    checkout = tmp_path / "checkout"
    (checkout / "deploy").mkdir(parents=True)
    (checkout / "deploy" / "brain-system.md").write_text("# d-brain session contract\n")
    vault = tmp_path / "elsewhere" / "vault"
    vault.mkdir(parents=True)

    s = Settings(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=vault,
        runtime_dir=tmp_path / "rt",
        project_root=checkout,
        _env_file=None,
    )
    assert s.project_root == checkout
    assert s.project_root != s.vault_path.parent  # the whole point
    sess = rt.get_session(s)
    assert sess.system_prompt_file == checkout / "deploy" / "brain-system.md"
    assert sess.work_dir == vault  # cwd is still the instance's own vault


def test_project_root_is_expanded_and_absolute(tmp_path):
    """Same absoluteness contract as vault_path/runtime_dir."""
    rt.reset()
    s = _settings(tmp_path, project_root="~")
    assert s.project_root is not None
    assert s.project_root.is_absolute()
    assert "~" not in str(s.project_root)
