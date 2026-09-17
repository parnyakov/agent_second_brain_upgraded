"""Contract test for the cron skill: the brain learns to manage its own
schedule from this file, so its key invariants are pinned here."""

from pathlib import Path

SKILL = Path(__file__).parent.parent / "vault/.claude/skills/cron/SKILL.md"


def test_skill_exists():
    assert SKILL.exists()


def test_skill_teaches_the_cli():
    text = SKILL.read_text()
    assert "python -m d_brain.cron" in text
    for cmd in ("add", "list", "remove", "enable"):
        assert cmd in text


def test_skill_pins_oneshot_and_silent_conventions():
    text = SKILL.read_text()
    assert "--delete-after-run" in text
    assert "[SILENT]" in text


def test_skill_forbids_sleep_emulation_and_recursion():
    text = SKILL.read_text().lower()
    assert "sleep" in text  # the "never emulate schedules with sleep" rule
    assert "cron job" in text  # the recursion-guard rule references the marker
