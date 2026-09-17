"""The whole point of v3.0: requests stay on the interactive subscription."""

import pathlib
import subprocess


def test_guard_passes():
    root = pathlib.Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        ["bash", str(root / "scripts" / "check-no-claude-p.sh")],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
