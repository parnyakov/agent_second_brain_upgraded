"""Suite-wide isolation of ``Settings`` from ambient configuration.

Why this file exists (2026-09-20). The production checkout keeps a real
``.env`` in the repo root, and ``Settings`` binds its dotenv path at CLASS
CREATION time::

    model_config = SettingsConfigDict(
        env_file=os.environ.get("DBRAIN_ENV_FILE", ".env"), ...)

So every ``Settings(**kwargs)`` in this suite silently inherits whatever that
file says for the fields the call does not pass. The same test then passes in
a clean git worktree (no ``.env`` there) and fails in the live checkout — the
exact split that hid four real failures of ``tests/test_chat_session.py``
behind a green run: the root ``.env`` still carried
``DBRAIN_CHAT_ENGINE=codex`` from the September Codex experiment, and the
turn-limit wording/health accounting is deliberately engine-dependent
(blind review F4 — Codex kills the turn on its deadline, Claude leaves it in
flight).

Two layers, because neither alone is enough:

1. ``env_file`` is rewritten to ``None`` ON THE CLASS. Setting
   ``DBRAIN_ENV_FILE`` from a fixture cannot help — the binding was read at
   import time, long before any fixture runs.
2. The engine/feature switches are dropped from the process environment, so
   ``DBRAIN_CHAT_ENGINE=codex uv run pytest`` is green too. A test that wants
   one of them sets it with ``monkeypatch.setenv``, which runs after this
   fixture and therefore still wins.

Deliberately NOT cleared: ``TZ`` and the credential-shaped variables
(``TELEGRAM_BOT_TOKEN`` & co). ``TZ`` is read by the C library as well as by
``Settings``, and unsetting it would quietly change time formatting across
the suite; the credential fields are passed explicitly by every helper that
builds a ``Settings``, and an explicit kwarg already outranks the
environment.
"""

import pytest

# Operator switches that change BEHAVIOR under test. Credentials and paths
# are left alone on purpose — see the module docstring.
_MANAGED_ENV = (
    "DBRAIN_ENV_FILE",
    "DBRAIN_CHAT_ENGINE",
    "DBRAIN_CRON_ENGINE",
    "DBRAIN_CODEX_MODEL",
    "DBRAIN_DUTY_SESSION",
    "DBRAIN_PANE_WIDTH",
    "DBRAIN_PANE_HEIGHT",
)


@pytest.fixture(autouse=True)
def isolate_settings_from_ambient_env(monkeypatch):
    """Make ``Settings`` depend only on what a test passes it."""
    from d_brain.config import Settings

    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in _MANAGED_ENV:
        monkeypatch.delenv(name, raising=False)
