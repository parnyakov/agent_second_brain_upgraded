# ruff: noqa: E501
"""setup.sh end-to-end, twice, against stubbed system tools.

Proves the promises the install guide makes: a first run asks for secrets
without echoing them, stores them owner-only, creates a PRIVATE vault
backup, applies the permission profile; a second run (interrupted install,
repeated bootstrap) asks for no secret again, keeps the owner's vault files
and remote, and does not duplicate managed configuration.

Only apt/npm/sudo/gh/systemctl/codex are stubbed; git, python3 and bash are
real. Network credential checks are skipped via DBRAIN_SETUP_OFFLINE_TEST.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

STUBS = {
    "sudo": r"""#!/bin/bash
echo "sudo $*" >> "$STUB_LOG"
[ "$1" = "-v" ] && exit 0
[ "$1" = "-n" ] && shift
case "$1" in
  apt-get|npm|loginctl) exit 0 ;;
  visudo) exit 0 ;;
  install|rm|test) "$@" ;;
  *) "$@" ;;
esac
""",
    "gh": r"""#!/bin/bash
echo "gh $*" >> "$STUB_LOG"
case "$1 $2" in
  "auth status"|"auth setup-git"|"auth login") exit 0 ;;
  "repo create")
    name="$3"; bare="$STUB_STATE/$name.git"
    git init -q --bare "$bare"
    git remote add origin "file://$bare"
    exit 0 ;;
  "repo view")
    target="$3"
    case "$target" in
      file://*) [ "$*" != "${*%visibility*}" ] && echo PRIVATE; exit 0 ;;
      *) [ -d "$STUB_STATE/$target.git" ] || exit 1
         case "$*" in *visibility*) echo PRIVATE ;; *url*) echo "file://$STUB_STATE/$target" ;; esac
         exit 0 ;;
    esac ;;
esac
exit 0
""",
    "codex": "#!/bin/bash\necho \"codex $*\" >> \"$STUB_LOG\"\n"
    "[ \"$1 $2\" = \"login status\" ] && echo 'Logged in using ChatGPT' >&2\nexit 0\n",
    "node": "#!/bin/bash\necho v20.11.0\n",
    "systemctl": "#!/bin/bash\necho \"systemctl $*\" >> \"$STUB_LOG\"\nexit 0\n",
    "timedatectl": "#!/bin/bash\necho Europe/Moscow\n",
    "tmux": "#!/bin/bash\nexit 0\n",
}

FAKE_TOKEN = "1234567" + ":" + "A" * 35
FAKE_DEEPGRAM = "abc123" * 5


def _project(tmp: Path) -> Path:
    project = tmp / "projects" / "agent-second-brain"
    project.mkdir(parents=True)
    for name in ("setup.sh", "bootstrap.sh", ".env.example", "pyproject.toml"):
        shutil.copy2(ROOT / name, project / name)
    shutil.copytree(ROOT / "templates", project / "templates")
    (project / "scripts").mkdir()
    for name in ("configure-permissions.sh", "setup_probe.py"):
        shutil.copy2(ROOT / "scripts" / name, project / "scripts" / name)
    (project / "src" / "d_brain").mkdir(parents=True)
    # The code repository ships vault/.claude (skills, persona).
    skill = project / "vault" / ".claude" / "skills" / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# demo\n")
    # The real upgrade.sh records the doctor's exit code; the stub replays
    # STUB_DOCTOR_RC (default green) so both endings of setup are testable.
    (project / "upgrade.sh").write_text(
        '#!/bin/bash\necho "upgrade.sh" >> "$STUB_LOG"\n'
        'mkdir -p "$HOME/.dbrain"\n'
        'echo "${STUB_DOCTOR_RC:-0}" > "$HOME/.dbrain/install-doctor.rc"\n'
    )
    return project


def _env(tmp: Path) -> dict[str, str]:
    stub_bin = tmp / "stub-bin"
    stub_bin.mkdir(exist_ok=True)
    for name, body in STUBS.items():
        path = stub_bin / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    home = tmp / "home"
    home.mkdir(exist_ok=True)
    (tmp / "state").mkdir(exist_ok=True)
    env = {
        "PATH": f"{stub_bin}:{os.environ['PATH']}",
        "HOME": str(home),
        "USER": "tester",
        "STUB_LOG": str(tmp / "stub.log"),
        "STUB_STATE": str(tmp / "state"),
        "DBRAIN_SETUP_OFFLINE_TEST": "1",
        "DBRAIN_SUDOERS_FILE": str(tmp / "sudoers.d" / "90-dbrain-tester"),
        "GIT_CONFIG_GLOBAL": str(tmp / "gitconfig"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "LANG": "C.UTF-8",
    }
    (tmp / "sudoers.d").mkdir(exist_ok=True)
    (tmp / "gitconfig").write_text("[init]\n\tdefaultBranch = main\n")
    return env


def _run(project: Path, env: dict[str, str], answers: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(project / "bootstrap.sh")],
        input="\n".join(answers) + "\n",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
def test_setup_first_run_then_rerun_is_idempotent(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("setup refuses root by design")
    project = _project(tmp_path)
    env = _env(tmp_path)

    # tz, engine, permissions, nightly cleanup, vault repo name: Enter
    # accepts every default.
    first = _run(project, env, [FAKE_TOKEN, "123456789", FAKE_DEEPGRAM, "", "", "", "", ""])
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    assert FAKE_TOKEN not in first.stdout + first.stderr
    assert FAKE_DEEPGRAM not in first.stdout + first.stderr

    env_file = project / ".env"
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    values = dict(
        line.split("=", 1) for line in env_file.read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )
    assert values["TELEGRAM_BOT_TOKEN"] == FAKE_TOKEN
    assert values["DEEPGRAM_API_KEY"] == FAKE_DEEPGRAM
    assert values["ALLOWED_USER_IDS"] == "[123456789]"
    assert values["DBRAIN_CHAT_ENGINE"] == values["DBRAIN_CRON_ENGINE"] == "codex"
    assert values["DBRAIN_PERMISSION_PROFILE"] == "max"
    assert values["DBRAIN_CODEX_SANDBOX"] == "danger-full-access"
    assert values["TZ"] == "Europe/Moscow"

    vault = project / "vault"
    assert (vault / "GUIDE.md").exists()
    assert (vault / "goals" / "3-weekly.md").exists()
    remote = subprocess.run(["git", "-C", str(vault), "remote", "get-url", "origin"],
                            capture_output=True, text=True, env=env).stdout.strip()
    assert remote.startswith("file://")
    tracked = subprocess.run(["git", "-C", str(vault), "ls-files"],
                             capture_output=True, text=True, env=env).stdout
    assert "GUIDE.md" in tracked
    assert ".claude/" not in tracked  # owned by the code repo, not by memory
    assert "DBRAIN_REQUIRE_PRIVATE_VAULT=1" in (project / ".env").read_text()
    sudoers = Path(env["DBRAIN_SUDOERS_FILE"])
    assert "NOPASSWD: ALL" in sudoers.read_text()
    claude = json.loads((tmp_path / "home" / ".claude" / "settings.json").read_text())
    assert claude["permissions"]["defaultMode"] == "bypassPermissions"

    # The owner writes into memory between runs; a rerun must keep it.
    (vault / "MEMORY.md").write_text("# Моя память\n\nважный факт\n")
    before_env = env_file.read_text()

    assert "Установка завершена" in first.stdout
    assert (tmp_path / "home" / ".dbrain" / "setup.completed").exists()
    # A fresh install must not get the old "keyboard removed" migration ping.
    assert (tmp_path / "home" / ".dbrain" / "keyboard_removed").exists()

    second = _run(project, env, ["", "", "", "", ""])
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-3000:]
    assert "Вставьте токен" not in second.stdout
    assert "Telegram ID (только цифры)" not in second.stdout
    assert "Имя приватного репозитория" not in second.stdout
    assert env_file.read_text().count("TELEGRAM_BOT_TOKEN=") == before_env.count("TELEGRAM_BOT_TOKEN=") == 1
    assert "важный факт" in (vault / "MEMORY.md").read_text()
    codex_config = (tmp_path / "home" / ".codex" / "config.toml").read_text()
    assert codex_config.count("# dbrain-permissions:begin") == 1
    log = (tmp_path / "stub.log").read_text()
    assert log.count("upgrade.sh") == 2
    assert log.count("gh repo create") == 1

    # Switching to the standard profile removes passwordless sudo.
    subprocess.run(["bash", str(project / "scripts" / "configure-permissions.sh"), "standard"],
                   env=env, check=True, capture_output=True, text=True)
    assert not sudoers.exists()
    assert "DBRAIN_CODEX_SANDBOX=workspace-write" in env_file.read_text()


def test_bootstrap_refuses_outside_a_distribution_checkout(tmp_path):
    lone = tmp_path / "bootstrap.sh"
    shutil.copy2(ROOT / "bootstrap.sh", lone)
    result = subprocess.run(["bash", str(lone)], capture_output=True, text=True, input="")
    assert result.returncode == 1
    assert "склонированного дистрибутива" in result.stdout


def test_setup_on_a_new_server_restores_memory_from_the_private_repo(tmp_path):
    """Lost server: a fresh install pointed at the existing memory repository
    must bring the saved vault back instead of failing on unrelated history."""
    if os.geteuid() == 0:
        pytest.skip("setup refuses root by design")
    old = _project(tmp_path / "old")
    env = _env(tmp_path)
    first = _run(old, env, [FAKE_TOKEN, "123456789", FAKE_DEEPGRAM, "", "", "", "", ""])
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    vault = old / "vault"
    (vault / "MEMORY.md").write_text("# Память\n\nсохранённый факт\n")
    subprocess.run(["git", "-C", str(vault), "commit", "-qam", "memory"], env=env, check=True)
    subprocess.run(["git", "-C", str(vault), "push", "-q"], env=env, check=True)

    new = _project(tmp_path / "new")
    second = _run(new, env, [FAKE_TOKEN, "123456789", FAKE_DEEPGRAM, "", "", "", "", "dbrain-vault"])
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-3000:]
    assert "Память восстановлена" in second.stdout
    assert "сохранённый факт" in (new / "vault" / "MEMORY.md").read_text()


def test_codex_not_logged_in_is_not_mistaken_for_logged_in(tmp_path):
    """Real codex CLI prints "Not logged in" (stderr, non-zero exit)."""
    stub = tmp_path / "bin"
    stub.mkdir()
    codex = stub / "codex"
    env = {"PATH": f"{stub}:{os.environ['PATH']}", "HOME": str(tmp_path)}
    check = f'source {ROOT / "setup.sh"}; codex_logged_in && echo YES || echo NO'
    for body, expected in (
        ("echo 'Not logged in' >&2; exit 1", "NO"),
        ("echo 'Not logged in' >&2; exit 0", "NO"),
        ("echo 'Logged in using ChatGPT' >&2; exit 0", "YES"),
    ):
        codex.write_text(f"#!/bin/bash\n{body}\n")
        codex.chmod(0o755)
        out = subprocess.run(["bash", "-c", check], capture_output=True, text=True, env=env)
        assert out.stdout.strip().endswith(expected), (body, out.stdout, out.stderr)


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
def test_setup_never_reports_success_when_the_agent_does_not_answer(tmp_path):
    # Clean-server rehearsal: the doctor's canary failed (engine not logged
    # in) but setup printed "[OK] Установка завершена" and announced it in
    # Telegram. Now it stops with the next action and a non-zero exit.
    if os.geteuid() == 0:
        pytest.skip("setup refuses root by design")
    project = _project(tmp_path)
    env = {**_env(tmp_path), "STUB_DOCTOR_RC": "1"}
    first = _run(project, env, [FAKE_TOKEN, "123456789", FAKE_DEEPGRAM, "", "", "", "", ""])
    assert first.returncode == 1
    assert "Установка завершена" not in first.stdout
    assert "агент пока не отвечает" in first.stdout
    assert "dbrain login" in first.stdout and "bootstrap.sh" in first.stdout
    assert not (tmp_path / "home" / ".dbrain" / "setup.completed").exists()

    # After the login is fixed, the documented rerun completes.
    env["STUB_DOCTOR_RC"] = "0"
    second = _run(project, env, ["", "", "", "", ""])
    assert second.returncode == 0, second.stdout[-3000:]
    assert "Установка завершена" in second.stdout
    assert "Вставьте токен" not in second.stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
def test_upgrade_exit_3_without_a_doctor_result_is_a_failure(tmp_path):
    # Exit 3 means "installed, health check red" only together with the
    # result file; any other exit 3 is a broken upgrade.
    if os.geteuid() == 0:
        pytest.skip("setup refuses root by design")
    project = _project(tmp_path)
    (project / "upgrade.sh").write_text('#!/bin/bash\nexit 3\n')
    env = _env(tmp_path)
    result = _run(project, env, [FAKE_TOKEN, "123456789", FAKE_DEEPGRAM, "", "", "", "", ""])
    assert result.returncode == 1
    assert "upgrade.sh завершился ошибкой" in result.stdout
    assert "Установка завершена" not in result.stdout
