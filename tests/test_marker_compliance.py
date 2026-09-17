"""Tests for scripts/marker_compliance.py's default transcript resolution.

R1 (Fable audit) pinned a runtime_dir/session_id file precisely to close a
footgun this script had: mtime-based "newest .jsonl" guessing can pick a
DIFFERENT concurrent session's file (the cron brain shares the exact same
project directory — see runtime.py). default_transcript_path() must now
prefer the pinned id, falling back to the old mtime guess (with a loud
warning) only when no pin is available.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "marker_compliance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "marker_compliance_under_test", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mc(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setattr(module.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("DBRAIN_RUNTIME_DIR", raising=False)
    return module, tmp_path


def _candidate_dir(module, tmp_path) -> Path:
    vault_path = _SCRIPT.resolve().parent.parent / "vault"
    slug = str(vault_path).replace("/", "-")
    return tmp_path / ".claude" / "projects" / slug


def test_prefers_pinned_session_id_when_present(mc):
    module, tmp_path = mc
    candidate_dir = _candidate_dir(module, tmp_path)
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "pinned-id.jsonl").touch()
    (candidate_dir / "other-concurrent-session.jsonl").touch()
    (tmp_path / ".dbrain").mkdir()
    (tmp_path / ".dbrain" / "session_id").write_text("pinned-id\n")

    result = module.default_transcript_path()
    assert result.name == "pinned-id.jsonl"


def test_falls_back_to_mtime_guess_when_pinned_transcript_missing(mc, capsys):
    module, tmp_path = mc
    candidate_dir = _candidate_dir(module, tmp_path)
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "only-one.jsonl").touch()
    (tmp_path / ".dbrain").mkdir()
    (tmp_path / ".dbrain" / "session_id").write_text("nonexistent-id\n")

    result = module.default_transcript_path()
    assert result.name == "only-one.jsonl"
    assert "no matching transcript" in capsys.readouterr().err


def test_falls_back_to_mtime_guess_with_ambiguity_warning_when_no_pin(mc, capsys):
    module, tmp_path = mc
    candidate_dir = _candidate_dir(module, tmp_path)
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "older.jsonl").touch()
    (candidate_dir / "newer.jsonl").touch()
    # No session_id file at all — pre-R1 install.

    result = module.default_transcript_path()
    assert result.name in {"older.jsonl", "newer.jsonl"}
    assert "may be a DIFFERENT concurrent session" in capsys.readouterr().err


def test_no_warning_when_single_transcript_and_no_pin(mc, capsys):
    module, tmp_path = mc
    candidate_dir = _candidate_dir(module, tmp_path)
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "only-one.jsonl").touch()

    result = module.default_transcript_path()
    assert result.name == "only-one.jsonl"
    assert capsys.readouterr().err == ""


def test_raises_when_no_transcripts_at_all(mc):
    module, tmp_path = mc
    candidate_dir = _candidate_dir(module, tmp_path)
    candidate_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        module.default_transcript_path()
