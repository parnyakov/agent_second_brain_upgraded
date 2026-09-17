"""delivery-watch.sh must not red-alert a server that has just started.

Found by the clean-server rehearsal: the timer's first run landed in the same
second as the watchdog's start, so every brand-new install sent the owner
"канал доставки под вопросом — watchdog ни разу не тикнул" at once.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run(
    tmp_path: Path, watchdog_since: str, restarts: str = "0"
) -> subprocess.CompletedProcess:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "delivery-watch.sh", project / "scripts")
    for name in ("notify.sh", "backup-notify.sh"):
        stub = project / "scripts" / name
        stub.write_text('#!/bin/bash\necho "$1" >> "$ALERTS"\n')
        stub.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    systemctl = bin_dir / "systemctl"
    systemctl.write_text(
        '#!/bin/bash\ncase "$*" in\n'
        '  *is-active*) exit 0 ;;\n'
        f'  *ActiveEnterTimestamp*) echo "{watchdog_since}" ;;\n'
        f'  *NRestarts*) echo "{restarts}" ;;\n'
        "esac\nexit 0\n"
    )
    systemctl.chmod(0o755)
    # Hermetic: the host's own bot processes must not count as orphans.
    pgrep = bin_dir / "pgrep"
    pgrep.write_text("#!/bin/bash\necho 1\n")
    pgrep.chmod(0o755)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DBRAIN_RUNTIME_DIR": str(runtime),
        "ALERTS": str(tmp_path / "alerts.txt"),
    }
    return subprocess.run(
        ["bash", str(project / "scripts" / "delivery-watch.sh")],
        env=env, capture_output=True, text=True, timeout=30,
    )


@pytest.mark.skipif(shutil.which("date") is None, reason="GNU date required")
def test_missing_status_right_after_start_is_not_a_fault(tmp_path):
    just_now = subprocess.run(
        ["date", "-d", "-20 seconds"], capture_output=True, text=True
    ).stdout.strip()
    result = _run(tmp_path, just_now)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "alerts.txt").exists()


def test_missing_status_long_after_start_is_still_a_fault(tmp_path):
    long_ago = subprocess.run(
        ["date", "-d", "-2 hours"], capture_output=True, text=True
    ).stdout.strip()
    result = _run(tmp_path, long_ago)
    assert result.returncode == 1
    assert "watchdog ни разу не тикнул" in (tmp_path / "alerts.txt").read_text()


def test_restarting_watchdog_gets_no_grace(tmp_path):
    just_now = subprocess.run(
        ["date", "-d", "-20 seconds"], capture_output=True, text=True
    ).stdout.strip()
    result = _run(tmp_path, just_now, restarts="4")
    assert result.returncode == 1
    assert "watchdog ни разу не тикнул" in (tmp_path / "alerts.txt").read_text()


def test_unknown_start_time_gets_no_grace(tmp_path):
    result = _run(tmp_path, "")
    assert result.returncode == 1
