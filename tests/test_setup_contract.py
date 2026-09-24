"""Contract tests for the install/upgrade scripts.

setup.sh once generated the dead v2 layout (system-level d-brain-* units)
— a fresh install was broken. These pins keep the install path honest:
setup.sh = interactive questions only, upgrade.sh = the single source of
truth for services and health.
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent
SETUP = (ROOT / "setup.sh").read_text()
UPGRADE = (ROOT / "upgrade.sh").read_text()


def test_setup_delegates_services_to_upgrade():
    assert "upgrade.sh" in SETUP


def test_setup_has_no_dead_v2_layout():
    assert "d-brain-" not in SETUP  # legacy unit names
    assert "/etc/systemd/system" not in SETUP  # v3 uses systemd --user


def test_setup_login_check_uses_json_not_prose():
    # `claude auth status | grep "Logged in"` broke when the CLI changed
    # its prose; the JSON field is the stable contract.
    assert "loggedIn" in SETUP


def test_setup_asks_timezone():
    assert "env_set TZ" in SETUP


def test_setup_supports_both_engines_and_hands_off_to_onboarding():
    assert "install_codex_cli" in SETUP
    assert "install_claude_cli" in SETUP
    # Onboarding is led by the agent in Telegram (voice answers, resumable);
    # the terminal wizard stays available through `dbrain onboarding`.
    assert "/onboarding" in SETUP
    assert "dbrain onboarding resume" in SETUP
    assert "initialize_vault" in SETUP
    assert 'templates/vault' in SETUP
    assert 'copy_missing "$PROJECT_DIR/templates/vault" "$VAULT_DIR"' in SETUP
    # coreutils 9.2+ warns on `cp -n`; the guide's clean output must not show it.
    assert 'cp -a -n' not in SETUP


def test_setup_never_puts_github_token_in_remote_url():
    assert "GITHUB_TOKEN@" not in SETUP
    assert "gh repo create" in SETUP
    assert "--private" in SETUP


def test_private_vault_check_fails_closed():
    assert 'if [ "$VISIBILITY" != "PRIVATE" ]' in SETUP
    assert "sync refused" in SETUP
    assert "NOPASSWD: ALL" not in SETUP


PERMISSIONS = (ROOT / "scripts" / "configure-permissions.sh").read_text()
BOOTSTRAP = (ROOT / "bootstrap.sh").read_text()


def test_passwordless_sudo_is_an_explicit_validated_profile():
    # Only the opt-in `max` profile writes sudoers, always validated first,
    # and `standard` removes it again.
    assert "visudo -cf" in PERMISSIONS
    assert "install -m 0440" in PERMISSIONS
    assert "disable_sudo" in PERMISSIONS and "standard)" in PERMISSIONS
    assert "choose_permissions" in SETUP


def test_setup_never_echoes_or_logs_secrets():
    assert "read -r -s" in SETUP  # secrets are typed without echo
    assert "setup_probe.py" in SETUP  # checks pass secrets via environment
    assert "set -x" not in SETUP


def test_bootstrap_never_downloads_code_or_points_at_a_fork():
    for text in (SETUP, BOOTSTRAP):
        assert "raw.githubusercontent.com" not in text
        assert "fork" not in text.lower()
    assert "curl" not in BOOTSTRAP


def test_upgrade_targets_v3():
    assert "v3.1" not in UPGRADE
    assert "dbrain-bot.service" in UPGRADE


def test_legacy_mac_installer_is_gone():
    assert not (ROOT / "install.sh").exists()


def test_setup_uses_the_services_claude_config_dir():
    # Clean-server rehearsal: login/first-run done without it never reached
    # the bot's session (units set CLAUDE_CONFIG_DIR=%h/.claude).
    fixed = 'export CLAUDE_CONFIG_DIR="$HOME/.claude"'
    assert fixed in SETUP
    assert SETUP.index(fixed) < SETUP.index("authorize_claude() {")
    export = 'export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"'
    assert export not in SETUP  # setup pins it; upgrade/dbrain keep an instance's own
    units = ("dbrain-bot.service", "dbrain-watchdog.service", "dbrain-doctor.service")
    for unit in units:
        text = (ROOT / "deploy" / unit).read_text()
        assert "Environment=CLAUDE_CONFIG_DIR=%h/.claude" in text
    assert export in (ROOT / "upgrade.sh").read_text()
    assert export in (ROOT / "bin" / "dbrain").read_text()
    # An existing login without the first run in that directory still opens
    # the TUI once, otherwise the bot's session stops on the first-run screens.
    body = SETUP[SETUP.index("authorize_claude() {"):]
    assert "claude_logged_in && claude_first_run_done" in body
    assert '"$CLAUDE_CONFIG_DIR/.claude.json"' in SETUP


# ── nightly cleanup is opt-in ──────────────────────────────────────────────
#
# The cleanup timer closes idle terminal windows. That is exactly the kind of
# thing a user must agree to, so the contract is: setup.sh asks, the answer
# lands in .env, and upgrade.sh arms the timer ONLY for an explicit "on".


def test_setup_asks_about_the_nightly_cleanup():
    assert "choose_cleanup" in SETUP
    assert "DBRAIN_CLEANUP" in SETUP


def test_cleanup_timer_follows_the_answer_and_defaults_to_off():
    # An install made before the question exists has no DBRAIN_CLEANUP line;
    # the empty value must take the disable branch, never the enable one.
    enable = UPGRADE.index("systemctl --user enable --now dbrain-cleanup.timer")
    guard = UPGRADE.index('if [ "$cleanup_choice" = "on" ]')
    assert guard < enable
    assert "disable --now dbrain-cleanup.timer" in UPGRADE


def test_cleanup_units_ship_with_the_project():
    for unit in ("dbrain-cleanup.service", "dbrain-cleanup.timer"):
        assert (ROOT / "deploy" / unit).exists()
    service = (ROOT / "deploy/dbrain-cleanup.service").read_text()
    # Same path placeholder upgrade.sh rewrites for every other unit.
    assert "%h/projects/dbrain" in service
