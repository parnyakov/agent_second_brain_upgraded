# ruff: noqa: E501
import json
import os
from datetime import date
from pathlib import Path

import pytest

import d_brain.onboarding as ob

BOT_TOKEN = "1234567890:" + "A" * 20 + "b" * 15


def _vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


def _cli(vault: Path, *args: str, capsys=None):
    code = ob.main(["--vault", str(vault), *args])
    out = capsys.readouterr() if capsys else None
    return code, out


def _answer_all_required(vault: Path, section_id: str) -> None:
    for question in ob.get_section(section_id).questions:
        if question.required:
            ob.save_answer(vault, section_id, question.id, _VALID.get(question.id, f"ответ {question.id}"))


_VALID = {"approved": "да", "timezone": "Europe/Moscow"}


def _finish_before_review(vault: Path) -> None:
    for name in ob.SECTIONS[: ob.SECTIONS.index("review")]:
        _answer_all_required(vault, name)
        ob.complete_section(vault, name)


# ── state basics ──────────────────────────────────────────────────────────


def test_state_is_versioned_and_owner_only(tmp_path: Path):
    vault = _vault(tmp_path)
    ob.save_answer(vault, "profile", "name", "Тест")
    path = ob.state_path(vault)
    raw = json.loads(path.read_text())
    assert raw["version"] == ob.STATE_VERSION == 2
    assert raw["sections"]["profile"]["answers"] == {"name": "Тест"}
    assert raw["sections"]["profile"]["status"] == "in_progress"
    assert path.stat().st_mode & 0o077 == 0
    assert not list(path.parent.glob("*.tmp"))


def test_unknown_state_version_fails_closed(tmp_path: Path):
    vault = _vault(tmp_path)
    ob.state_path(vault).parent.mkdir()
    ob.state_path(vault).write_text('{"version": 999}')
    with pytest.raises(ob.OnboardingError, match="Версия состояния"):
        ob.load_state(vault)


def test_v1_state_is_migrated_with_backup(tmp_path: Path):
    vault = _vault(tmp_path)
    path = ob.state_path(vault)
    path.parent.mkdir()
    v1 = {
        "version": 1,
        "completed": ["profile", "work", "goals"],
        "answers": {
            "profile": {"name": "Тест", "timezone": "Europe/Moscow", "language": "Русский", "role": "Менеджер", "about": ""},
            "work": {"projects": "Проект А", "people": "", "responsibilities": "Продажи"},
            "goals": {"vision": "", "year": "Рост", "month": "Запуск", "week": "Отчёт"},
            "notes": {"deferred": True},
        },
        "updated_at": "2026-09-01T10:00:00+03:00",
    }
    path.write_text(json.dumps(v1, ensure_ascii=False))

    state = ob.load_state(vault)

    assert state["version"] == 2
    assert state["sections"]["profile"]["status"] == "done"
    assert state["sections"]["profile"]["answers"]["name"] == "Тест"
    assert "about" not in state["sections"]["profile"]["answers"]
    assert state["sections"]["projects"]["status"] == "done"
    assert state["sections"]["projects"]["answers"]["responsibilities"] == "Продажи"
    assert state["sections"]["goals"]["answers"] == {"year": "Рост", "month": "Запуск", "week": "Отчёт"}
    assert state["sections"]["notes"]["status"] == "deferred"
    assert state["sections"]["preferences"]["status"] == "pending"
    backup = path.with_name("state.v1.backup.json")
    assert json.loads(backup.read_text()) == v1
    assert json.loads(path.read_text())["version"] == 2
    assert ob.next_section(state) == "preferences"


# ── engine: resume, ordering, reopen ──────────────────────────────────────


def test_resume_mid_section_at_first_unanswered_question(tmp_path: Path):
    vault = _vault(tmp_path)
    ob.save_answer(vault, "profile", "name", "Тест")
    ob.save_answer(vault, "profile", "timezone", "Europe/Moscow")

    payload = ob.next_payload(ob.load_state(vault))

    assert payload["section"]["id"] == "profile"
    assert payload["section"]["first_unanswered"] == "language"
    assert payload["section"]["answers"] == {"name": "Тест", "timezone": "Europe/Moscow"}


def test_optional_empty_answer_counts_as_answered(tmp_path: Path):
    vault = _vault(tmp_path)
    ob.save_answer(vault, "goals", "vision_3y", "")
    assert ob.first_unanswered(ob.load_state(vault), "goals") == "year"
    with pytest.raises(ob.OnboardingError, match="обязательный"):
        ob.save_answer(vault, "goals", "year", "   ")


def test_deferred_sections_come_after_all_pending(tmp_path: Path):
    vault = _vault(tmp_path)
    ob.defer_section(vault, "profile")
    ob.defer_section(vault, "goals")
    state = ob.load_state(vault)
    assert ob.next_section(state) == "projects"

    ob.save_answer(vault, "services", "used", "Gmail")
    assert ob.next_section(ob.load_state(vault)) == "services"  # in_progress wins

    for name in ob.SECTIONS:
        if name in ("profile", "goals"):
            continue
        _answer_all_required(vault, name)
        ob.complete_section(vault, name)
    assert ob.next_section(ob.load_state(vault)) == "profile"
    _answer_all_required(vault, "profile")
    ob.complete_section(vault, "profile")
    assert ob.next_section(ob.load_state(vault)) == "goals"
    _answer_all_required(vault, "goals")
    ob.complete_section(vault, "goals")
    assert ob.next_section(ob.load_state(vault)) is None


def test_reopen_affects_only_that_section(tmp_path: Path):
    vault = _vault(tmp_path)
    for name in ("profile", "projects", "goals"):
        _answer_all_required(vault, name)
        ob.complete_section(vault, name)

    ob.reopen_section(vault, "projects")
    state = ob.load_state(vault)

    assert state["sections"]["profile"]["status"] == "done"
    assert state["sections"]["goals"]["status"] == "done"
    assert state["sections"]["projects"]["status"] == "in_progress"
    assert state["sections"]["projects"]["answers"]["projects"] == "ответ projects"
    assert ob.next_section(state) == "projects"


def test_complete_refuses_missing_required(tmp_path: Path):
    vault = _vault(tmp_path)
    ob.save_answer(vault, "profile", "name", "Тест")
    with pytest.raises(ob.OnboardingError, match="timezone"):
        ob.complete_section(vault, "profile")
    assert ob.load_state(vault)["sections"]["profile"]["status"] == "in_progress"
    assert not (vault / "personal" / "about.md").exists()


def test_review_requires_explicit_approval(tmp_path: Path):
    vault = _vault(tmp_path)
    _finish_before_review(vault)
    ob.save_answer(vault, "review", "approved", "нет, цели неверные")
    with pytest.raises(ob.OnboardingError, match="подтверждения"):
        ob.complete_section(vault, "review")
    ob.save_answer(vault, "review", "approved", "Да")
    _, written = ob.complete_section(vault, "review")
    assert written == [".onboarding/review.md"]
    assert "Подтверждено" in (vault / ".onboarding" / "review.md").read_text()


def test_review_requires_earlier_sections_done_or_deferred(tmp_path: Path):
    vault = _vault(tmp_path)
    _answer_all_required(vault, "profile")
    ob.complete_section(vault, "profile")
    ob.defer_section(vault, "notes")
    _answer_all_required(vault, "review")
    with pytest.raises(ob.OnboardingError) as exc:
        ob.complete_section(vault, "review")
    message = str(exc.value)
    for name in ("projects", "goals", "preferences", "services"):
        assert f"({name})" in message
    assert "(profile)" not in message and "(notes)" not in message
    assert ob.load_state(vault)["sections"]["review"]["status"] == "in_progress"
    assert not (vault / ".onboarding" / "review.md").exists()


def test_capabilities_requires_review_done(tmp_path: Path, capsys):
    vault = _vault(tmp_path)
    for name in ob.SECTIONS[: ob.SECTIONS.index("review")]:
        ob.defer_section(vault, name)
    code, out = _cli(vault, "complete", "capabilities", capsys=capsys)
    assert code == 2 and "(review)" in out.err
    _answer_all_required(vault, "review")
    ob.complete_section(vault, "review")  # deferred earlier sections are allowed
    assert _cli(vault, "complete", "capabilities", capsys=capsys)[0] == 0


@pytest.mark.parametrize("zone", ["Mars/Olympus", "Moscow", "../etc/passwd", "UTC+3"])
def test_invalid_timezone_refused(tmp_path: Path, zone: str, capsys):
    vault = _vault(tmp_path)
    code, out = _cli(vault, "answer", "profile", "timezone", "--value", zone, capsys=capsys)
    assert code == 2
    assert "Europe/Moscow" in out.err
    assert "timezone" not in ob.load_state(vault)["sections"]["profile"]["answers"]


def test_valid_timezone_accepted(tmp_path: Path):
    vault = _vault(tmp_path)
    for zone in ("Europe/Moscow", "Asia/Almaty", "UTC"):
        ob.save_answer(vault, "profile", "timezone", zone)
    assert ob.load_state(vault)["sections"]["profile"]["answers"]["timezone"] == "UTC"


# ── rendering ─────────────────────────────────────────────────────────────


def test_managed_block_preserves_outside_edits(tmp_path: Path):
    vault = _vault(tmp_path)
    _answer_all_required(vault, "profile")
    ob.complete_section(vault, "profile")
    about = vault / "personal" / "about.md"
    about.write_text("Заметка сверху\n\n" + about.read_text() + "\nЗаметка снизу\n")

    ob.save_answer(vault, "profile", "role", "Директор")
    ob.complete_section(vault, "profile")

    text = about.read_text()
    assert text.startswith("Заметка сверху")
    assert text.rstrip().endswith("Заметка снизу")
    assert "Директор" in text and "ответ role" not in text
    assert text.count(ob.BLOCK_BEGIN) == 1 and text.count(ob.BLOCK_END) == 1


def test_markerless_template_gets_block_appended(tmp_path: Path):
    vault = _vault(tmp_path)
    template = Path(__file__).parent.parent / "templates" / "vault"
    (vault / "goals").mkdir()
    for name in ("0-vision-3y.md", "1-yearly.md", "2-monthly.md", "3-weekly.md"):
        (vault / "goals" / name).write_text((template / "goals" / name).read_text())
    (vault / "MEMORY.md").write_text((template / "MEMORY.md").read_text())
    original_memory = (vault / "MEMORY.md").read_text()

    _answer_all_required(vault, "goals")
    ob.save_answer(vault, "goals", "week", "Сдать отчёт")
    _, written = ob.complete_section(vault, "goals", today=date(2026, 9, 16))
    _answer_all_required(vault, "preferences")
    ob.complete_section(vault, "preferences")

    assert "goals/1-yearly.md" in written
    yearly = (vault / "goals" / "1-yearly.md").read_text()
    assert yearly.startswith("# Цели года")
    assert "2026" in yearly and ob.BLOCK_BEGIN in yearly
    assert "Сдать отчёт" in (vault / "goals" / "3-weekly.md").read_text()
    memory = (vault / "MEMORY.md").read_text()
    assert memory.startswith(original_memory.rstrip())
    assert "ответ style" in memory
    assert not list((vault / "goals").glob("1-yearly-*.md"))


@pytest.mark.parametrize(
    ("given", "canonical", "sample"),
    [("Мужской", "мужской", "«сделал»"), ("ж", "женский", "«сделала»"), ("нейтрально", "нейтрально", "«готово»")],
)
def test_agent_persona_rendered_into_memory(tmp_path: Path, given, canonical, sample):
    vault = _vault(tmp_path)
    (vault / "MEMORY.md").write_text("# Долговременная память\n\nСвоё.\n")
    _answer_all_required(vault, "preferences")
    ob.save_answer(vault, "preferences", "agent_name", "Помощник")
    ob.save_answer(vault, "preferences", "agent_gender", given)
    assert ob.load_state(vault)["sections"]["preferences"]["answers"]["agent_gender"] == canonical
    ob.complete_section(vault, "preferences")
    memory = (vault / "MEMORY.md").read_text()
    block = memory[memory.index(ob.BLOCK_BEGIN): memory.index(ob.BLOCK_END)]
    assert "## Агент" in block
    assert "- Имя: Помощник" in block
    assert f"- Как говорить о себе: {canonical}" in block and sample in block
    assert "Своё." in memory


def test_agent_persona_optional_defaults_to_neutral(tmp_path: Path):
    vault = _vault(tmp_path)
    _answer_all_required(vault, "preferences")
    ob.complete_section(vault, "preferences")
    memory = (vault / "MEMORY.md").read_text()
    assert "- Имя: не задано" in memory
    assert "- Как говорить о себе: нейтрально" in memory
    with pytest.raises(ob.OnboardingError, match="мужской"):
        ob.save_answer(vault, "preferences", "agent_gender", "средний")


def test_backups_and_archives_never_keep_secret_answers(tmp_path: Path, capsys):
    vault = _vault(tmp_path)
    path = ob.state_path(vault)
    path.parent.mkdir()
    v1 = {"version": 1, "completed": [], "answers": {"services": {"used": f"бот {BOT_TOKEN}", "wanted": "Gmail"}}}
    path.write_text(json.dumps(v1, ensure_ascii=False))
    ob.load_state(vault)
    ob.save_answer(vault, "profile", "name", "Тест")
    code, out = _cli(vault, "restart", "--yes", "--json", capsys=capsys)
    assert code == 0
    for file in path.parent.iterdir():
        text = file.read_text()
        assert BOT_TOKEN not in text, file.name
    backup = json.loads(path.with_name("state.v1.backup.json").read_text())
    assert backup["answers"]["services"] == {"used": ob.REDACTED, "wanted": "Gmail"}
    assert "Тест" in Path(json.loads(out.out)["archived"]).read_text()


def test_services_and_notes_targets(tmp_path: Path):
    vault = _vault(tmp_path)
    _finish_before_review(vault)
    _answer_all_required(vault, "review")
    ob.complete_section(vault, "review")
    ob.save_answer(vault, "capabilities", "questions", "")
    _, written = ob.complete_section(vault, "capabilities")
    assert ".env" in (vault / "personal" / "connections.md").read_text()
    readme = (vault / "inbox" / "README.md").read_text()
    assert "raw-import" in readme and "import-plan.md" in readme
    assert written == []
    assert ob.load_state(vault)["sections"]["capabilities"]["guide_shown_at"]


# ── secrets ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value",
    [
        f"мой токен {BOT_TOKEN}",
        "github_pat_11ABCDEF",
        "ghp_abcdefghijklmnop",
        "sk-proj-abcdefghijk",
        "-----BEGIN OPENSSH " + "PRIVATE KEY-----",
        "ключ a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8",
    ],
)
def test_secret_like_answer_refused_and_not_persisted(tmp_path: Path, value: str, capsys):
    vault = _vault(tmp_path)
    code, out = _cli(vault, "answer", "services", "used", "--value", value, "--json", capsys=capsys)
    assert code == 2
    assert ".env" in out.err
    assert json.loads(out.out)["ok"] is False
    state_file = ob.state_path(vault)
    assert not state_file.exists() or value not in state_file.read_text()


def test_ordinary_text_is_not_a_secret():
    for text in ("Google Drive, Notion, task-list", "Europe/Moscow", "https://example.com/risk-assessment"):
        assert not ob.looks_like_secret(text)


# ── notes import ──────────────────────────────────────────────────────────


def test_import_idempotent_manifest_source_untouched_symlink_skipped(tmp_path: Path):
    vault = _vault(tmp_path)
    source = tmp_path / "export"
    (source / "sub").mkdir(parents=True)
    (source / "one.md").write_text("первая", encoding="utf-8")
    (source / "sub" / "two.html").write_text("<p>вторая</p>", encoding="utf-8")
    (source / "skip.pdf").write_bytes(b"pdf")
    (source / "big.txt").write_bytes(b"x" * (ob.MAX_IMPORT_BYTES + 1))
    outside = tmp_path / "secret.md"
    outside.write_text("чужое")
    os.symlink(outside, source / "link.md")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.rglob("*") if p.is_file() and not p.is_symlink()}

    today = date(2026, 9, 16)
    first = ob.import_notes(vault, source, today=today)
    second = ob.import_notes(vault, source, today=today)

    assert first["copied"] == 2 and first["unsupported"] == 1
    assert {s["reason"] for s in first["skipped"]} == {"symlink", "too_large"}
    assert second["copied"] == 0 and second["duplicates"] == 2
    raw = vault / "inbox" / "raw-import" / "2026-09-16"
    assert (raw / "one.md").read_text() == "первая"
    assert (raw / "sub" / "two.html").exists()
    assert not (raw / "link.md").exists() and not (raw / "big.txt").exists()
    manifest = ob.load_state(vault)["sections"]["notes"]["manifest"]
    assert len(manifest) == 2
    assert {entry["dest"] for entry in manifest.values()} == {
        "inbox/raw-import/2026-09-16/one.md",
        "inbox/raw-import/2026-09-16/sub/two.html",
    }
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.rglob("*") if p.is_file() and not p.is_symlink()}
    assert before == after
    assert outside.read_text() == "чужое"


def test_import_notes_cli_json(tmp_path: Path, capsys):
    vault = _vault(tmp_path)
    note = tmp_path / "note.txt"
    note.write_text("текст")
    code, out = _cli(vault, "import-notes", str(note), "--json", capsys=capsys)
    payload = json.loads(out.out)
    assert code == 0 and payload["copied"] == 1
    code, out = _cli(vault, "import-notes", str(tmp_path / "missing"), capsys=capsys)
    assert code == 2 and "не найден" in out.err


# ── CLI schema ────────────────────────────────────────────────────────────


def test_status_and_next_json_schema(tmp_path: Path, capsys):
    vault = _vault(tmp_path)
    code, out = _cli(vault, "status", "--json", capsys=capsys)
    status = json.loads(out.out)
    assert code == 0
    assert {"version", "done", "total", "finished", "next_section", "sections", "updated_at"} <= set(status)
    assert status["total"] == 8 and status["next_section"] == "profile"
    assert [s["id"] for s in status["sections"]] == list(ob.SECTIONS)
    assert {"id", "index", "title", "status", "answered", "total_questions", "missing_required", "updated_at"} <= set(status["sections"][0])

    code, out = _cli(vault, "answer", "profile", "name", "--value", "Тест", "--json", capsys=capsys)
    assert code == 0 and json.loads(out.out)["first_unanswered"] == "timezone"

    code, out = _cli(vault, "next", "--json", capsys=capsys)
    nxt = json.loads(out.out)
    assert nxt["finished"] is False
    section = nxt["section"]
    assert {"id", "title", "why", "status", "questions", "answers", "first_unanswered", "missing_required"} <= set(section)
    assert section["why"] and section["first_unanswered"] == "timezone"
    assert {"id", "prompt", "hint", "required", "multiline", "answer"} <= set(section["questions"][0])


def test_goals_cover_vision_year_month_week():
    ids = [q.id for q in ob.get_section("goals").questions]
    assert ids == ["vision_3y", "year", "month", "week"]
    assert not ob.get_section("goals").question("vision_3y").required


def test_cli_usage_errors_exit_2(tmp_path: Path, capsys):
    vault = _vault(tmp_path)
    assert _cli(vault, "answer", "nope", "x", "--value", "1", capsys=capsys)[0] == 2
    assert _cli(vault, "answer", "profile", "nope", "--value", "1", capsys=capsys)[0] == 2
    assert _cli(vault, "restart", capsys=capsys)[0] == 2
    code, out = _cli(vault, "complete", "profile", capsys=capsys)
    assert code == 2 and "обязательных" in out.err
    with pytest.raises(SystemExit) as exc:
        ob.main(["--vault", str(vault), "answer", "profile", "name"])
    assert exc.value.code == 2


def test_answer_from_stdin(tmp_path: Path, monkeypatch, capsys):
    import io

    vault = _vault(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("строка 1\nстрока 2\n"))
    code, _ = _cli(vault, "answer", "profile", "about", "--stdin", capsys=capsys)
    assert code == 0
    assert ob.load_state(vault)["sections"]["profile"]["answers"]["about"] == "строка 1\nстрока 2"


# ── wizard & restart ──────────────────────────────────────────────────────


def _scripted(lines: list[str]):
    feed = iter(lines)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    return fake_input


def test_wizard_eof_saves_partial_answers_and_resumes(tmp_path: Path, capsys):
    vault = _vault(tmp_path)
    code = ob.run_wizard(vault, input_fn=_scripted(["Тест", "Europe/Moscow"]))
    assert code == 130
    assert "dbrain onboarding resume" in capsys.readouterr().out
    state = ob.load_state(vault)
    assert state["sections"]["profile"]["answers"] == {"name": "Тест", "timezone": "Europe/Moscow"}
    assert state["sections"]["profile"]["status"] == "in_progress"

    # Resume: language, role, about (multiline, "." ends), then Ctrl+C in projects.
    def interrupt_after(lines):
        inner = _scripted(lines)

        def fake(prompt=""):
            try:
                return inner(prompt)
            except EOFError:
                raise KeyboardInterrupt from None

        return fake

    code = ob.run_wizard(vault, input_fn=interrupt_after(["русский", "Менеджер", "Строка", "."]))
    assert code == 130
    state = ob.load_state(vault)
    assert state["sections"]["profile"]["status"] == "done"
    assert state["sections"]["profile"]["answers"]["about"] == "Строка"
    assert (vault / "personal" / "about.md").exists()
    assert ob.next_section(state) == "projects"


def test_wizard_refuses_secret_and_reasks(tmp_path: Path, capsys):
    vault = _vault(tmp_path)
    ob.defer_section(vault, "profile")
    for name in ob.SECTIONS[1:4]:
        ob.defer_section(vault, name)
    code = ob.run_wizard(vault, input_fn=_scripted([BOT_TOKEN, ".", "Gmail", "."]))
    assert code == 130
    assert ".env" in capsys.readouterr().out
    state_text = ob.state_path(vault).read_text()
    assert BOT_TOKEN not in state_text
    assert ob.load_state(vault)["sections"]["services"]["answers"]["used"] == "Gmail"


def test_restart_archives_state(tmp_path: Path, capsys):
    vault = _vault(tmp_path)
    ob.save_answer(vault, "profile", "name", "Тест")
    code, out = _cli(vault, "restart", "--yes", "--json", capsys=capsys)
    assert code == 0
    archived = Path(json.loads(out.out)["archived"])
    assert archived.exists() and "Тест" in archived.read_text()
    assert archived.name.startswith("state.2")
    fresh = ob.load_state(vault)
    assert fresh["sections"]["profile"]["answers"] == {}
    assert ob.restart(vault) != archived  # second restart in the same second does not collide


def test_clean_template_has_no_personal_content():
    root = Path(__file__).parent.parent
    template = root / "templates" / "vault"
    names = {str(path.relative_to(template)) for path in template.rglob("*") if path.is_file()}
    assert not any(name.startswith("daily/20") for name in names)
    corpus = "\n".join(path.read_text(errors="ignore") for path in template.rglob("*") if path.is_file())
    # Personal terms live only in the maintainer's denylist (never shipped);
    # the export privacy gate scans the template too.
    denylist = root / "distribution" / "denylist.txt"
    if denylist.exists():
        import re

        for line in denylist.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                assert not re.search(line.strip(), corpus, re.I), line


def test_daily_log_masks_pasted_credentials(tmp_path):
    """A token pasted into chat during onboarding must not reach the synced
    daily log (the onboarding state already refuses it)."""
    from datetime import datetime

    from d_brain.services.storage import SECRET_PLACEHOLDER, VaultStorage

    token = "1234567" + ":" + "Ab" * 20
    pat = "github" + "_pat_" + "X1" * 15
    storage = VaultStorage(tmp_path / "vault")
    stamp = datetime(2026, 1, 1, 12, 0)
    storage.append_to_daily(f"мой токен {token} и {pat}, цель: 10 клиентов", stamp, "[text]")
    content = storage.read_daily(stamp.date())
    assert token not in content and pat not in content
    assert content.count(SECRET_PLACEHOLDER) == 2
    assert "цель: 10 клиентов" in content
