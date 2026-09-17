"""Tests for the daily self-diagnostic (Doctor)."""

from d_brain.services.claude_session import AskResult
from d_brain.services.doctor import CheckResult, Doctor


class FakeSession:
    def __init__(self, result: AskResult) -> None:
        self.result = result

    def ask(self, prompt, *, timeout=120, request_id=None) -> AskResult:  # noqa: ANN001
        return self.result


def _ok_check():
    return CheckResult("disk", True, "6 GB free")


def _bad_check():
    return CheckResult("git", False, "push failed")


def test_canary_ok_makes_report_ok():
    sess = FakeSession(AskResult("ok", reply="DBRAIN_OK"))
    rep = Doctor(sess, checks=[_ok_check]).run()
    assert rep.ok
    assert any(c.name == "canary" and c.ok for c in rep.checks)


def test_canary_wrong_reply_fails():
    sess = FakeSession(AskResult("ok", reply="hello?"))
    rep = Doctor(sess, checks=[]).run()
    assert not rep.ok


def test_canary_logged_out_fails_with_reason():
    sess = FakeSession(AskResult("logged_out"))
    rep = Doctor(sess, checks=[]).run()
    canary = next(c for c in rep.checks if c.name == "canary")
    assert not canary.ok
    assert "вход" in canary.detail.lower() or "log" in canary.detail.lower()


def test_failing_local_check_makes_report_not_ok():
    sess = FakeSession(AskResult("ok", reply="DBRAIN_OK"))
    rep = Doctor(sess, checks=[_bad_check]).run()
    assert not rep.ok


def test_report_telegram_green_and_red():
    sess = FakeSession(AskResult("ok", reply="DBRAIN_OK"))
    green = Doctor(sess, checks=[_ok_check]).run().to_telegram()
    assert "🟢" in green
    red = Doctor(sess, checks=[_bad_check]).run().to_telegram()
    assert "🔴" in red
    assert "❌" in red


def test_run_cli_exit_codes_follow_report():
    # upgrade.sh and the systemd OnFailure= hook key off the exit code —
    # a failing canary must be visible as a non-zero exit.
    from d_brain.services.doctor import run_cli

    sent = []
    ok_sess = FakeSession(AskResult("ok", reply="DBRAIN_OK"))
    assert run_cli(ok_sess, checks=[_ok_check], alert=sent.append) == 0

    bad_sess = FakeSession(AskResult("logged_out"))
    assert run_cli(bad_sess, checks=[], alert=sent.append) == 1
    assert len(sent) == 2  # the telegram report goes out either way


def test_canary_is_tagged_as_maintenance():
    # Chat steering keys off the maint- prefix in the inflight id — the
    # canary must never look like a steerable user turn.
    class RecordingSession(FakeSession):
        def __init__(self, result):
            super().__init__(result)
            self.request_ids = []

        def ask(self, prompt, *, timeout=120, request_id=None):
            self.request_ids.append(request_id)
            return self.result

    sess = RecordingSession(AskResult("ok", reply="DBRAIN_OK"))
    Doctor(sess, checks=[]).run()
    assert sess.request_ids and sess.request_ids[0].startswith("maint-")


def test_check_claude_version_resolves_binary_outside_path(monkeypatch, tmp_path):
    # Manual runs (ssh, cron) often lack ~/.local/bin in PATH — the check
    # must resolve the binary like the services do, not false-alarm.
    import d_brain.services.doctor as doc

    fake_bin = tmp_path / ".local" / "bin" / "claude"
    fake_bin.parent.mkdir(parents=True)
    fake_bin.write_text("#!/bin/sh\necho 2.1.0\n")
    fake_bin.chmod(0o755)

    # An EMPTY directory, not a real system PATH: on a box that happens to
    # have `claude` installed under /usr/bin (true for this dev container),
    # "/usr/bin:/bin" does not exercise "no ~/.local/bin in PATH" at all —
    # shutil.which("claude") finds the REAL binary before ever falling
    # through to the home-dir fallback this test means to cover, and the
    # test then silently asserts against that real binary's version instead
    # of the fake one, breaking hermeticity (review 2026-08-20 T2 gate).
    empty_path_dir = tmp_path / "empty-path"
    empty_path_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_path_dir))
    monkeypatch.setattr(doc.Path, "home", staticmethod(lambda: tmp_path))

    res = doc.check_claude_version()
    assert res.ok
    assert "2.1.0" in res.detail


# ── R7 (Fable audit): weekly marker-compliance trend check ────────────────


class _FakeSessionWithTranscript:
    def __init__(self, path):
        self._path = path

    def current_transcript_path(self):
        return self._path


def test_check_marker_compliance_not_due_yet_skips(tmp_path):
    from d_brain.services.doctor import check_marker_compliance

    (tmp_path / "last_marker_compliance_check").write_text("1000\n")
    res = check_marker_compliance(
        _FakeSessionWithTranscript(None),
        tmp_path,
        repo_dir=tmp_path,
        min_interval_seconds=999_999,
        clock_fn=lambda: 1001.0,
    )
    assert res.ok is True
    assert "не пора" in res.detail


def test_check_marker_compliance_due_but_no_transcript(tmp_path):
    from d_brain.services.doctor import check_marker_compliance

    res = check_marker_compliance(
        _FakeSessionWithTranscript(None),
        tmp_path,
        repo_dir=tmp_path,
        min_interval_seconds=1.0,
        clock_fn=lambda: 1_000_000.0,
    )
    assert res.ok is True
    assert "нет транскрипта" in res.detail


def test_check_marker_compliance_runs_script_and_persists_last_run(tmp_path):
    from d_brain.services.doctor import check_marker_compliance

    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "marker_compliance.py").write_text(
        "print('total marker-wrap turns: 42')\n"
    )
    res = check_marker_compliance(
        _FakeSessionWithTranscript(transcript),
        tmp_path,
        repo_dir=tmp_path,
        min_interval_seconds=1.0,
        clock_fn=lambda: 1_000_000.0,
    )
    assert res.ok is True  # a trend metric, never fails the daily doctor
    assert "42" in res.detail
    persisted = (tmp_path / "last_marker_compliance_check").read_text().strip()
    assert persisted == "1000000.0"


def test_check_marker_compliance_never_crashes_doctor_on_a_bad_session(tmp_path):
    from d_brain.services.doctor import check_marker_compliance

    class BrokenSession:
        def current_transcript_path(self):
            raise RuntimeError("boom")

    res = check_marker_compliance(
        BrokenSession(), tmp_path, repo_dir=tmp_path, min_interval_seconds=1.0,
        clock_fn=lambda: 1_000_000.0,
    )
    assert res.ok is True
    assert "пропущено" in res.detail


# ── clean-server rehearsal: Codex installs must not be judged by `claude` ──


def test_engine_check_uses_the_configured_engine(monkeypatch, tmp_path):
    import d_brain.services.doctor as doc

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    codex = bin_dir / "codex"
    codex.write_text("#!/bin/sh\necho codex-cli 0.154.0\n")
    codex.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(doc.Path, "home", staticmethod(lambda: tmp_path))

    res = doc.check_engine_version("codex")
    assert res.ok and res.name == "codex" and "0.154.0" in res.detail
    missing = doc.check_engine_version("claude")
    assert not missing.ok and missing.name == "claude" and missing.hint


def test_logged_out_names_the_engine_and_the_next_action():
    sess = FakeSession(AskResult("logged_out"))
    rep = Doctor(sess, checks=[], engine="codex").run()
    canary = rep.checks[0]
    assert "Codex" in canary.detail and "Claude" not in canary.detail
    assert "dbrain login" in canary.hint
    assert "dbrain login" in rep.to_telegram()


def test_run_cli_prints_the_checklist_for_the_terminal(capsys):
    from d_brain.services.doctor import run_cli

    bad = FakeSession(AskResult("logged_out"))
    assert run_cli(bad, checks=[_ok_check], alert=lambda _m: None, engine="codex") == 1
    printed = capsys.readouterr().out
    assert "Осмотр: есть проблемы" in printed
    assert "❌ canary" in printed and "✅ disk" in printed
    assert "→ на сервере выполните dbrain login" in printed


def test_install_canary_retries_a_slow_first_answer_but_not_a_logout():
    class Sequence:
        def __init__(self, results):
            self.results, self.calls = list(results), 0

        def ask(self, prompt, *, timeout=120, request_id=None):  # noqa: ANN001
            self.calls += 1
            return self.results.pop(0)

    slow = Sequence(
        [AskResult("timeout", detail="no answer"), AskResult("ok", reply="DBRAIN_OK")]
    )
    naps = []
    rep = Doctor(slow, checks=[], canary_attempts=2, sleep=naps.append).run()
    assert rep.ok and slow.calls == 2 and len(naps) == 1

    logged_out = Sequence([AskResult("logged_out"), AskResult("ok", reply="DBRAIN_OK")])
    rep = Doctor(logged_out, checks=[], canary_attempts=2, sleep=naps.append).run()
    assert not rep.ok and logged_out.calls == 1


def test_telegram_report_is_plain_text():
    # The watchdog alerter posts without parse_mode: tags would show literally.
    sess = FakeSession(AskResult("logged_out"))
    text = Doctor(sess, checks=[_bad_check], engine="codex").run().to_telegram()
    assert "<" not in text and ">" not in text.replace("→", "")
