"""Memory pushes stay fail-closed after install, not only during setup."""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path

import pytest

from d_brain.services import git as vault_git


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "vault"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin",
         "https://github.com/example/dbrain-vault.git"],
        check=True,
    )
    return repo


def _fake_gh(tmp_path: Path, monkeypatch, visibility: str | None) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    if visibility is None:
        gh.write_text("#!/bin/sh\nexit 1\n")
    else:
        gh.write_text(f'#!/bin/sh\necho \'{{"visibility":"{visibility}"}}\'\n')
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")


@pytest.mark.parametrize(
    ("visibility", "allowed"),
    [("PRIVATE", True), ("PUBLIC", False), ("INTERNAL", False), (None, False)],
)
def test_push_requires_private_remote(tmp_path, monkeypatch, visibility, allowed):
    repo = _repo(tmp_path)
    _fake_gh(tmp_path, monkeypatch, visibility)
    monkeypatch.setenv(vault_git.REQUIRE_PRIVATE_ENV, "1")
    assert vault_git.main(["check-private", str(repo)]) == (0 if allowed else 1)

    pushed = []
    monkeypatch.setattr(
        vault_git.VaultGit, "_run_git",
        lambda self, *args: pushed.append(args)
        or subprocess.CompletedProcess(args, 0, "", ""),
    )
    assert vault_git.VaultGit(repo).push() is allowed
    assert bool(pushed) is allowed


def test_guard_is_inactive_without_the_install_flag(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _fake_gh(tmp_path, monkeypatch, "PUBLIC")
    monkeypatch.delenv(vault_git.REQUIRE_PRIVATE_ENV, raising=False)
    assert vault_git.main(["check-private", str(repo)]) == 0


def test_nightly_script_checks_before_push():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "process.sh").read_text()
    body = script.split("_commit_repo() {", 1)[1].split("\n}", 1)[0]
    assert body.index("check-private") < body.index('push || true')


def test_process_push_reads_the_flag_from_the_install_env_file(tmp_path, monkeypatch):
    """The bot does not export .env: the flag written by setup into
    <project>/.env must still stop /process from pushing to a PUBLIC repo."""
    repo = _repo(tmp_path)
    _fake_gh(tmp_path, monkeypatch, "PUBLIC")
    monkeypatch.delenv(vault_git.REQUIRE_PRIVATE_ENV, raising=False)
    (tmp_path / ".env").write_text("TZ=UTC\nDBRAIN_REQUIRE_PRIVATE_VAULT=1\n")
    pushed = []
    monkeypatch.setattr(
        vault_git.VaultGit, "_run_git",
        lambda self, *args: pushed.append(args)
        or subprocess.CompletedProcess(args, 0, "", ""),
    )
    assert vault_git.VaultGit(repo).push() is False
    assert pushed == []
