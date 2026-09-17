"""upgrade.sh records the install-time doctor result that setup.sh gates on.

setup.sh reports "Установка завершена" only when this file says 0, so the
real upgrade.sh (not a stub) must write it and map a red doctor to exit 3.
Every system command is stubbed in $HOME/.local/bin, which upgrade.sh puts
first on PATH, so the test never touches the host's apt, sudo or systemd.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

STUB = '#!/bin/bash\necho "$(basename "$0") $*" >> "$STUB_LOG"\n'
STUBS = {
    "sudo": STUB + "exit 1\n",
    "apt-get": STUB + "exit 0\n",
    "git": STUB + "exit 0\n",
    "loginctl": STUB + "exit 0\n",
    "tmux": STUB + "exit 0\n",
    # is-enabled fails: the single-instance --user branch is taken.
    "systemctl": STUB + 'case "$*" in *is-enabled*) exit 1 ;; esac\nexit 0\n',
    "uv": STUB
    + 'case "$*" in *d_brain.services.doctor*)\n'
    + '  echo "attempts=$DBRAIN_DOCTOR_CANARY_ATTEMPTS" >> "$STUB_LOG"\n'
    + '  exit "$STUB_DOCTOR_RC" ;;\nesac\nexit 0\n',
}


def _run(tmp_path: Path, doctor_rc: int) -> tuple[subprocess.CompletedProcess, Path]:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "bin").mkdir()
    (project / "src").mkdir()
    shutil.copy2(ROOT / "upgrade.sh", project / "upgrade.sh")
    shutil.copy2(ROOT / "scripts" / "check-no-claude-p.sh", project / "scripts")
    shutil.copy2(ROOT / "bin" / "dbrain", project / "bin" / "dbrain")
    shutil.copytree(
        ROOT / "deploy", project / "deploy",
        ignore=shutil.ignore_patterns("incidents", "systemd", "*.md"),
    )
    home = tmp_path / "home"
    stubs = home / ".local" / "bin"
    stubs.mkdir(parents=True)
    for name, body in STUBS.items():
        (stubs / name).write_text(body)
        (stubs / name).chmod(0o755)
    env = {
        "HOME": str(home),
        "USER": "tester",
        "PATH": f"{stubs}:/usr/bin:/bin",
        "STUB_LOG": str(tmp_path / "stub.log"),
        "STUB_DOCTOR_RC": str(doctor_rc),
        "LANG": "C.UTF-8",
    }
    result = subprocess.run(["bash", str(project / "upgrade.sh")], env=env,
                            capture_output=True, text=True, timeout=60)
    return result, home / ".dbrain" / "install-doctor.rc"


@pytest.mark.skipif(os.name != "posix", reason="bash script")
def test_green_doctor_records_zero_and_completes(tmp_path):
    result, rc_file = _run(tmp_path, 0)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert rc_file.read_text().strip() == "0"
    assert "Upgrade complete" in result.stdout
    log = (tmp_path / "stub.log").read_text()
    assert "attempts=2" in log  # install-time canary may retry a slow first boot


@pytest.mark.skipif(os.name != "posix", reason="bash script")
def test_red_doctor_records_the_code_and_exits_3(tmp_path):
    result, rc_file = _run(tmp_path, 1)
    assert result.returncode == 3, result.stdout[-2000:] + result.stderr[-2000:]
    assert rc_file.read_text().strip() == "1"
    assert "Upgrade complete" not in result.stdout
    assert "dbrain doctor" in result.stdout
