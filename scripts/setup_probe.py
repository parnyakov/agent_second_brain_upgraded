#!/usr/bin/env python3
"""Live credential checks for setup.sh (standard library only, pre-uv).

Secrets arrive via the environment (TELEGRAM_BOT_TOKEN, DEEPGRAM_API_KEY,
CHAT_ID), never argv, and are never printed.

Exit codes: 0 ok, 3 Telegram token rejected, 4 the owner has not started the
bot yet (or the chat id is wrong), 5 Deepgram key rejected, 6 network error.

``--announce`` sends the "installation finished" message instead.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

TIMEOUT = 20


def telegram(method: str, token: str, **params: str) -> dict:
    data = urllib.parse.urlencode(params).encode() if params else None
    request = urllib.request.Request(f"https://api.telegram.org/bot{token}/{method}", data=data)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            return json.load(exc)
        except Exception:
            return {"ok": False, "error_code": exc.code}


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("CHAT_ID", "")
    try:
        if "--announce" in sys.argv:
            telegram(
                "sendMessage",
                token,
                chat_id=chat_id,
                text=(
                    "Установка завершена. Отправьте /onboarding: я задам вопросы, чтобы "
                    "понять ваши задачи и цели. Отвечать можно голосом, прерываться можно "
                    "в любой момент."
                ),
            )
            return 0
        me = telegram("getMe", token)
        if not me.get("ok"):
            print("Telegram отклонил токен бота")
            return 3
        username = me.get("result", {}).get("username", "")
        sent = telegram(
            "sendMessage",
            token,
            chat_id=chat_id,
            text="Проверка связи: установка агента идёт. Отвечать на это сообщение не нужно.",
        )
        if not sent.get("ok"):
            print(f"Бот @{username} не может написать вам: откройте бота и нажмите Start")
            return 4
        key = os.environ.get("DEEPGRAM_API_KEY", "")
        request = urllib.request.Request(
            "https://api.deepgram.com/v1/projects", headers={"Authorization": f"Token {key}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT):
                pass
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                print("Deepgram отклонил ключ")
                return 5
            raise
        print(f"Бот @{username} работает и написал вам в Telegram; ключ Deepgram принят")
        return 0
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"сетевая ошибка: {type(exc).__name__}")
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
