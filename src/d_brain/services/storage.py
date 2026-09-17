"""Vault storage service for saving entries."""

import re
from datetime import date, datetime
from pathlib import Path

# Credentials a user may paste into chat by mistake. The daily log is synced
# to git, so these are masked before they ever reach disk. Unambiguous
# signatures only: ordinary text must pass through unchanged.
_SECRET_PATTERNS = (
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}"),  # Telegram bot token
    re.compile(r"\b(?:github_pat_[A-Za-z0-9_]{20,}|gh[opsur]_[A-Za-z0-9]{30,})"),
    re.compile(r"\bsk-(?:ant-|proj-|svcacct-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\bGOCSPX-[0-9A-Za-z_-]{20,}"),
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"
        r".*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)",
        re.S,
    ),
)
SECRET_PLACEHOLDER = "[секрет скрыт: отзовите его и создайте новый]"


def mask_secrets(text: str) -> str:
    """Replace credential-looking substrings with a placeholder."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(SECRET_PLACEHOLDER, text)
    return text


class VaultStorage:
    """Service for storing entries in Obsidian vault."""

    def __init__(self, vault_path: Path) -> None:
        self.vault_path = Path(vault_path)
        self.daily_path = self.vault_path / "daily"
        self.attachments_path = self.vault_path / "attachments"

    def _ensure_dirs(self) -> None:
        """Ensure required directories exist."""
        self.daily_path.mkdir(parents=True, exist_ok=True)
        self.attachments_path.mkdir(parents=True, exist_ok=True)

    def get_daily_file(self, day: date) -> Path:
        """Get path to daily file for given date."""
        self._ensure_dirs()
        return self.daily_path / f"{day.isoformat()}.md"

    def read_daily(self, day: date) -> str:
        """Read content of daily file."""
        file_path = self.get_daily_file(day)
        if not file_path.exists():
            return ""
        return file_path.read_text(encoding="utf-8")

    def append_to_daily(
        self,
        text: str,
        timestamp: datetime,
        msg_type: str,
    ) -> None:
        """Append entry to daily file.

        Args:
            text: Content to append
            timestamp: Entry timestamp
            msg_type: Type marker like [voice], [text], [photo], [forward from: Name]
        """
        self._ensure_dirs()
        file_path = self.get_daily_file(timestamp.date())

        time_str = timestamp.strftime("%H:%M")
        entry = f"\n## {time_str} {msg_type}\n{mask_secrets(text)}\n"

        with file_path.open("a", encoding="utf-8") as f:
            f.write(entry)

    def get_attachments_dir(self, day: date) -> Path:
        """Get attachments directory for given date."""
        dir_path = self.attachments_path / day.isoformat()
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def save_attachment(
        self,
        data: bytes,
        day: date,
        timestamp: datetime,
        extension: str = "jpg",
    ) -> str:
        """Save attachment and return relative path for Obsidian embed.

        Args:
            data: File bytes
            day: Date for organizing
            timestamp: Timestamp for filename
            extension: File extension

        Returns:
            Relative path for Obsidian embed: attachments/YYYY-MM-DD/img-HHMMSS.ext
        """
        dir_path = self.get_attachments_dir(day)
        time_str = timestamp.strftime("%H%M%S")
        # Telegram albums arrive as N messages within the same second —
        # uniquify so concurrent saves never overwrite each other.
        filename = f"img-{time_str}.{extension}"
        counter = 1
        while (dir_path / filename).exists():
            filename = f"img-{time_str}-{counter}.{extension}"
            counter += 1
        file_path = dir_path / filename

        file_path.write_bytes(data)

        return f"attachments/{day.isoformat()}/{filename}"
