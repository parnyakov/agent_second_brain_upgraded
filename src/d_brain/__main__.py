"""Entry point for running d-brain as a module."""

import asyncio
import logging

from d_brain import logsafe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logsafe.install()

# httpx logs every request URL at INFO, and a Telegram Bot API URL
# carries the bot token — keep it out of the journal / log files.
for _name in ("httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    from d_brain.bot.main import run_bot
    from d_brain.config import get_settings

    settings = get_settings()
    logger.info("d-brain starting...")
    logger.info("Vault path: %s", settings.vault_path)
    logger.info("Allowed users: %s", settings.allowed_user_ids or "all")

    await run_bot(settings)


if __name__ == "__main__":
    asyncio.run(main())
