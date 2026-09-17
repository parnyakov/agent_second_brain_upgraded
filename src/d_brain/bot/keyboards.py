"""Reply keyboards for Telegram bot."""

from aiogram.types import ReplyKeyboardMarkup
from aiogram.utils.keyboard import ReplyKeyboardBuilder


def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Main reply keyboard with common commands."""
    builder = ReplyKeyboardBuilder()
    builder.button(text="📊 Статус")
    builder.button(text="⚙️ Обработать")
    builder.button(text="❓ Помощь")
    builder.adjust(3)
    return builder.as_markup(resize_keyboard=True, is_persistent=True)
