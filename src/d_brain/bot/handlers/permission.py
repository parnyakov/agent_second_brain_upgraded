"""«Да» / «Нет» on a permission question (services/permission_gate.py).

The handler only unpacks the button: who pressed it, which dialog it was
for, yes or no. Whether it still applies, and the key that answers the
CLI, is the gate's business. Every outcome is said: the toast on the button
and the question edited to show what happened.
"""

import logging
from contextlib import suppress

from aiogram import F, Router
from aiogram.types import CallbackQuery

from d_brain.services import permission_gate

router = Router(name="permission")
logger = logging.getLogger(__name__)


@router.callback_query(F.data.startswith(permission_gate.CALLBACK_PREFIX))
async def handle_permission_answer(callback: CallbackQuery) -> None:
    gate = permission_gate.current()
    parsed = permission_gate.parse_callback(callback.data or "")
    if gate is None or parsed is None:
        await callback.answer("Кнопка не сработала: запрос не найден.")
        return
    slot, nonce, allow = parsed
    # An old message Telegram no longer lets us touch has neither.
    message = callback.message
    editable = callable(getattr(message, "edit_text", None))
    toast, text = await gate.answer(
        slot,
        nonce,
        allow,
        user_id=callback.from_user.id if callback.from_user else None,
        current_html=(getattr(message, "html_text", "") or "") if editable else "",
    )
    with suppress(Exception):
        await callback.answer(toast)
    if text is not None and editable:
        try:
            await message.edit_text(text, reply_markup=None)
        except Exception:  # noqa: BLE001 — the answer itself is already applied
            logger.warning("permission: could not edit the question", exc_info=True)
