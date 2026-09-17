"""Git automation service for vault."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRE_PRIVATE_ENV = "DBRAIN_REQUIRE_PRIVATE_VAULT"


def private_remote_required(vault: Path | None = None) -> bool:
    """Installs made by setup.sh require the memory remote to stay PRIVATE.

    The flag lives in the install's ``.env`` (next to ``vault/``). The bot
    process does not export ``.env`` into its environment, so the file is
    read directly; the environment variable also counts.
    """
    values = [os.environ.get(REQUIRE_PRIVATE_ENV, "")]
    if vault is not None:
        env_file = Path(vault).resolve().parent / ".env"
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith(f"{REQUIRE_PRIVATE_ENV}="):
                    values.append(line.split("=", 1)[1])
        except OSError:
            pass
    return any(v.strip().strip("'\"").lower() in {"1", "true", "yes"} for v in values)


def remote_visibility(repo: Path) -> str:
    """GitHub visibility of ``origin`` ("PRIVATE", "PUBLIC", ...) or "UNKNOWN".

    Asked on every push, not only at install time: a repository made public
    later must stop receiving memory. Any failure to prove PRIVATE (no gh,
    no network, not GitHub) is reported as UNKNOWN, and callers fail closed.
    """
    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo, capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        if not url:
            return "UNKNOWN"
        out = subprocess.run(
            ["gh", "repo", "view", url, "--json", "visibility"],
            cwd=repo, capture_output=True, text=True, check=True, timeout=60,
        ).stdout
        return str(json.loads(out).get("visibility", "UNKNOWN")).upper()
    except (OSError, subprocess.SubprocessError, ValueError):
        return "UNKNOWN"


class VaultGit:
    """Service for git operations on vault."""

    def __init__(self, vault_path: Path) -> None:
        self.vault_path = Path(vault_path)

    def _run_git(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run git command in vault directory."""
        return subprocess.run(
            ["git", *args],
            cwd=self.vault_path,
            capture_output=True,
            text=True,
            check=False,
        )

    def get_status(self) -> str:
        """Get git status."""
        result = self._run_git("status", "--porcelain")
        return result.stdout

    def has_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        return bool(self.get_status().strip())

    def commit_changes(self, message: str) -> bool:
        """Stage all changes and commit.

        Args:
            message: Commit message

        Returns:
            True if commit was made, False otherwise
        """
        if not self.has_changes():
            logger.info("No changes to commit")
            return False

        # Stage all changes
        add_result = self._run_git("add", "-A")
        if add_result.returncode != 0:
            logger.error("Git add failed: %s", add_result.stderr)
            return False

        # Commit
        commit_result = self._run_git("commit", "-m", message)
        if commit_result.returncode != 0:
            logger.error("Git commit failed: %s", commit_result.stderr)
            return False

        logger.info("Committed: %s", message)
        return True

    def push(self) -> bool:
        """Push to remote.

        Returns:
            True if push was successful
        """
        if private_remote_required(self.vault_path):
            visibility = remote_visibility(self.vault_path)
            if visibility != "PRIVATE":
                logger.error(
                    "Push refused: memory remote is %s, not PRIVATE", visibility
                )
                return False
        result = self._run_git("push")
        if result.returncode != 0:
            logger.error("Git push failed: %s", result.stderr)
            return False

        logger.info("Pushed to remote")
        return True

    def commit_and_push(self, message: str) -> bool:
        """Commit all changes and push.

        Args:
            message: Commit message

        Returns:
            True if successful
        """
        if self.commit_changes(message):
            return self.push()
        return True  # No changes is not an error


def main(argv: list[str] | None = None) -> int:
    """``python -m d_brain.services.git check-private <repo>``: exit 0 only if
    the push may proceed (not required, or the remote is PRIVATE)."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] != "check-private":
        usage = "usage: python -m d_brain.services.git check-private <repo>"
        print(usage, file=sys.stderr)
        return 2
    if not private_remote_required(Path(args[1])):
        return 0
    visibility = remote_visibility(Path(args[1]))
    if visibility == "PRIVATE":
        return 0
    print(f"memory remote is {visibility}, not PRIVATE; push refused", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
