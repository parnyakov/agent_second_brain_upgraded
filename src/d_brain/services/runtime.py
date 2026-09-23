"""Process-wide singletons for the shared Claude session.

The bot, the daily pipeline and the watchdog must all talk to ONE persistent
session. This module builds it lazily from Settings and hands the same
instance to every caller. An asyncio lock serializes ask() calls within the
bot process (the cross-process flock in ClaudeSession is the real mutex; this
just avoids piling up blocked worker threads).
"""

import asyncio
import uuid
from pathlib import Path

from d_brain.config import Settings
from d_brain.services.claude_session import ClaudeSession
from d_brain.services.codex_driver import CodexExecDriver
from d_brain.services.engine import EngineDriver
from d_brain.services.processor import ClaudeProcessor

# Typed as EngineDriver (the Protocol in engine.py) rather than ClaudeSession:
# this is the seam the Codex pilot flips. Purely a typing widening — the only
# object ever constructed today is still ClaudeSession.
_session: EngineDriver | None = None
_cron_session: EngineDriver | None = None
_duty_session: EngineDriver | None = None
_processor: ClaudeProcessor | None = None
_ask_lock = asyncio.Lock()


def reset() -> None:
    """Drop the singletons (tests only)."""
    global _session, _cron_session, _duty_session, _processor
    _session = None
    _cron_session = None
    _duty_session = None
    _processor = None


def _persisted_name(settings: Settings) -> str:
    if settings.brain_session_name:
        return settings.brain_session_name
    # Randomize per install (fingerprint hygiene) and persist so restarts
    # reuse the same tmux session name.
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    name_file = settings.runtime_dir / "brain.name"
    if name_file.exists():
        return name_file.read_text().strip()
    name = f"dbrain_{uuid.uuid4().hex[:8]}"
    name_file.write_text(name + "\n")
    return name


def _stall_kwarg(stall_timeout: float | None) -> dict[str, float]:
    """``{"stall_timeout": x}`` or ``{}`` (review round 3, R1).

    An EMPTY dict when unset, rather than passing the engine constant
    explicitly: the main and cron sessions must keep constructing their
    driver with byte-for-byte the arguments they did before this parameter
    existed, so that whatever each driver's own default is — including a
    future change to it — remains theirs. Only the duty session, which is
    watched by nobody (see Settings.duty_stall_timeout), opts into an
    override.
    """
    return {} if stall_timeout is None else {"stall_timeout": float(stall_timeout)}


def _build_codex_session(
    settings: Settings,
    *,
    session_name: str,
    runtime_dir: Path,
    stall_timeout: float | None = None,
) -> EngineDriver:
    """Construct the Codex-backed driver (Codex plan, phase 2).

    Mirrors the claude branch below argument-for-argument where the concept
    survives — same session_name, same vault as work_dir, same per-session
    runtime_dir, same model setting — and drops what has no Codex meaning:

    * ``mcp_config`` / ``tmux_config`` — MCP is configured in Codex's own
      config, and there is no tmux.
    * ``system_prompt_file`` → ``instructions_file``. Claude Code takes the
      persona via ``--append-system-prompt``; Codex takes AGENTS.md-shaped
      standing instructions, so this points at ``deploy/codex-agents.md``,
      the Codex counterpart of ``deploy/brain-system.md``.

    Keeps the claude branch's boot assertion, for the same reason: without a
    persona file the brain silently starts as a vanilla agent with no
    identity and no reply contract. Refuse loudly instead.
    """
    project_root = settings.project_root or settings.vault_path.parent
    instructions = project_root / "deploy" / "codex-agents.md"
    if (
        not instructions.exists()
        or "# d-brain codex agent contract" not in instructions.read_text()
    ):
        raise RuntimeError(
            f"codex persona file missing or invalid: {instructions} — "
            "refusing to start a personality-less brain"
        )
    # Only an explicitly configured sandbox is passed; unset keeps the
    # driver's own default for existing installs.
    extra = {"sandbox": settings.codex_sandbox} if settings.codex_sandbox else {}
    return CodexExecDriver(
        session_name=session_name,
        work_dir=settings.vault_path,
        runtime_dir=runtime_dir,
        instructions_file=instructions,
        model=settings.codex_model or None,
        **extra,
        **_stall_kwarg(stall_timeout),
    )


def _build_session(
    settings: Settings,
    *,
    session_name: str,
    runtime_dir: Path,
    engine: str = "claude",
    stall_timeout: float | None = None,
) -> EngineDriver:
    # Engine seam (Codex plan, phase 1). Everything below this branch is the
    # unmodified pre-seam function: engine="claude" — the default, and the
    # only value production ever passes today — builds byte-for-byte the same
    # ClaudeSession with the same arguments as before this parameter existed.
    #
    # `stall_timeout` (review round 3, R1) keeps that property: left unset it
    # reaches neither driver's constructor at all — see _stall_kwarg.
    if engine == "codex":
        return _build_codex_session(
            settings,
            session_name=session_name,
            runtime_dir=runtime_dir,
            stall_timeout=stall_timeout,
        )
    if engine != "claude":
        raise ValueError(f"unknown engine: {engine!r}")
    # C2: was `settings.vault_path.parent` inline. Settings fills project_root
    # with exactly that when PROJECT_ROOT is unset, so this is a no-op for the
    # single-instance install; a second instance whose vault lives outside the
    # code checkout points it at the checkout instead.
    # Settings._default_project_root always fills this in. The `or` is a real
    # fallback rather than an `assert`, which `python -O` strips — leaving a
    # None to surface later as a confusing TypeError on a path join.
    project_root = settings.project_root or settings.vault_path.parent
    mcp = project_root / "mcp-config.json"
    brain_prompt = project_root / "deploy" / "brain-system.md"
    # B4 fix (Fable audit fix round, 2026-08-22): passed through so
    # ClaudeSession can apply it via `-f` at `new-session` time — see
    # tmux_config's docstring in claude_session.py for why the `-g
    # set-option` path alone isn't enough on a cold tmux server.
    tmux_conf = project_root / "deploy" / "tmux.conf"
    # Boot assertion: without the persona file the brain would silently start
    # as a vanilla agent (no identity, no reply contract). Refuse loudly.
    if (
        not brain_prompt.exists()
        or "# d-brain session contract" not in brain_prompt.read_text()
    ):
        raise RuntimeError(
            f"persona file missing or invalid: {brain_prompt} — "
            "refusing to start a personality-less brain"
        )
    return ClaudeSession(
        session_name=session_name,
        work_dir=settings.vault_path,
        runtime_dir=runtime_dir,
        mcp_config=mcp if mcp.exists() else None,
        system_prompt_file=brain_prompt,
        model=settings.claude_model or None,
        tmux_config=tmux_conf if tmux_conf.exists() else None,
        # Defaults equal the previously hardcoded 200x50, so an install
        # that sets neither env var is unchanged (see Settings.pane_width).
        pane_width=settings.pane_width,
        pane_height=settings.pane_height,
        **_stall_kwarg(stall_timeout),
    )


def get_session(settings: Settings) -> EngineDriver:
    """Return the shared interactive session singleton (ClaudeSession today)."""
    global _session
    if _session is None:
        _session = _build_session(
            settings,
            session_name=_persisted_name(settings),
            runtime_dir=settings.runtime_dir,
            engine=settings.chat_engine,
        )
    return _session


def get_cron_session(settings: Settings) -> EngineDriver:
    """Return the cron brain — a second, isolated ClaudeSession.

    Same persona and vault as the main brain, but its own tmux session and
    its own runtime dir (pane.lock / pane.log / ready), so scheduled jobs
    never block or pollute the user's conversation.
    """
    global _cron_session
    if _cron_session is None:
        _cron_session = _build_session(
            settings,
            session_name=f"{_persisted_name(settings)}_cron",
            runtime_dir=settings.cron_dir,
            engine=settings.cron_engine,
        )
    return _cron_session


def get_duty_session(settings: Settings) -> EngineDriver:
    """Return the duty brain — a THIRD, isolated session (backlog items 29-30).

    Exact sibling of ``get_cron_session``: same persona and vault, its own
    session name and its own runtime dir. Two things are deliberately
    different:

    * It follows ``chat_engine``, not ``cron_engine``. The duty session
      stands in for the CHAT brain when that one is busy, so flipping the
      chat engine to Codex must take the stand-in with it — otherwise an
      operator running Codex for chat would still boot a tmux Claude
      session behind their back.
    * It is never ``/clear``-ed per job the way the cron session is; the
      clear is time-based (``duty_idle_reset_seconds``), so several
      messages arriving inside one busy window keep their own thread.
    * It is the ONLY session built with an explicit ``stall_timeout``
      (review round 3, R1). The engine default (900s) sits above this
      session's whole turn budget (``duty_turn_timeout``, 600s), which made
      the stall interrupt unreachable — and ``duty_dir`` is watched by
      nobody: the watchdog polls ``settings.runtime_dir`` only, so neither
      its recovery nor its orphan-reply poller covers this session. Self-
      interruption inside the budget is the only backstop it has.

    The cron session is NOT reused for this on purpose: it is wiped after
    every scheduled job and can be busy with one when the user writes.
    """
    global _duty_session
    if _duty_session is None:
        _duty_session = _build_session(
            settings,
            session_name=f"{_persisted_name(settings)}_duty",
            runtime_dir=settings.duty_dir,
            engine=settings.chat_engine,
            stall_timeout=settings.duty_stall_timeout or None,
        )
    return _duty_session


def get_processor(settings: Settings) -> ClaudeProcessor:
    global _processor
    if _processor is None:
        _processor = ClaudeProcessor(
            settings.vault_path,
            session=get_session(settings),
        )
    return _processor


def get_ask_lock() -> asyncio.Lock:
    return _ask_lock
