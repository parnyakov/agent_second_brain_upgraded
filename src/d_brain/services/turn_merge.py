"""Bind follow-up input to the Telegram request that owns the chat lane."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar

# Propagated by asyncio.to_thread to both ask() and steer().
owner = ContextVar("chat_turn_owner", default=None)


class Opening:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.extra: list[str] = []
        self.started = asyncio.Event()
        self.closed = False
        self.cancelled = False
        self.sealed = False
        self.steer_lock = asyncio.Lock()

    def seal(self, prompt: str) -> str:
        self.sealed = True
        self.started.set()
        return "\n\n".join([prompt, *self.extra])

    async def wait_for_engine(self, manager) -> bool:
        deadline = asyncio.get_running_loop().time() + 20
        while not self.closed and asyncio.get_running_loop().time() < deadline:
            if manager.is_steerable_turn():
                return True
            await asyncio.sleep(0.05)
        return False

    async def join(self, manager, prompt: str) -> bool:
        if self.closed or self.cancelled:
            return False
        if not self.sealed:
            self.extra.append(prompt)
            await self.started.wait()
            return self.sealed and not self.cancelled
        # Serialize consecutive clarifications in arrival order, including
        # the small interval between submitting the prompt and turn/started.
        async with self.steer_lock:
            token = owner.set(self)
            try:
                if not self.cancelled and await self.wait_for_engine(manager):
                    return await manager.steer(prompt) is not False
            finally:
                owner.reset(token)
        return False


_current: dict[int, Opening] = {}


def current(chat_id: int, user_id: int) -> Opening | None:
    opening = _current.get(chat_id)
    return opening if opening and opening.user_id == user_id else None


@contextmanager
def register(chat_id: int, user_id: int):
    opening = Opening(user_id)
    _current.setdefault(chat_id, opening)
    token = owner.set(opening)
    try:
        yield opening
    finally:
        owner.reset(token)
        opening.closed = True
        opening.started.set()
        if _current.get(chat_id) is opening:
            del _current[chat_id]
