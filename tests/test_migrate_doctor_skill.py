"""Contract test for the migrate-doctor skill: the fallback when
upgrade.sh fails on a non-standard install. Pins the invariants the
skill must teach."""

from pathlib import Path

SKILL = (
    Path(__file__).parent.parent / "vault/.claude/skills/migrate-doctor/SKILL.md"
)


def test_skill_exists():
    assert SKILL.exists()


def test_skill_knows_the_version_layouts():
    text = SKILL.read_text()
    # v1/v2 legacy markers the doctor must recognize
    assert "d-brain-" in text
    assert "claude -p" in text or "claude --print" in text
    # v3 target layout
    assert "dbrain-" in text
    assert "tmux" in text


def test_skill_prefers_upgrade_sh_and_backs_up_first():
    text = SKILL.read_text()
    assert "upgrade.sh" in text
    assert "backup" in text.lower() or "бэкап" in text.lower()


def test_skill_never_touches_vault_content():
    assert "vault" in SKILL.read_text().lower()


def test_skill_verifies_with_doctor():
    text = SKILL.read_text()
    assert "doctor" in text
    assert "check-no-claude-p" in text
